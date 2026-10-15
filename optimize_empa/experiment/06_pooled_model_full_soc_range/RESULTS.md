# optimize_empa, experiment 06: one pooled model, full SOC range

## The question this answers

Every prior attempt (01-05) trained a fresh RF + XGBoost pair *per*
`Initial_SOC` bucket, because the synthetic generator never produced
data below `Initial_SOC=0.5`. Investigating why Random Forest had
collapsed to ~50% balanced accuracy (always predicting NMC) while
XGBoost scored 95-99.97% on the identical data traced the cause to
that per-SOC specialization itself: it shrank each bucket's training
set to ~100-500 rows, which `RandomForestClassifier(max_depth=10)`
overfits on (100% train accuracy, spreading importance across every
noisy bin) while XGBoost's regularization (`max_depth=4`,
`subsample=0.8`, `colsample_bytree=0.8`) happens to tolerate.

Separately, the actual production goal was never "one model per SOC
bucket" -- it's one model that predicts chemistry regardless of the
battery's *unknown* test-time SOC. This experiment abandons per-SOC
specialization entirely and goes back to the original "one model,
evaluated at every SOC" design (`optimize_empa/base_pipeline/evaluate_empa_soc_sweep.py`),
but fixes what was blocking it: the synthetic generator's
`Initial_SOC` floor at 0.5.

## What changed

**Data generation**: switched from `sim.solve(initial_soc=X)`
(REST-initialized -- a battery that sat idle at X% then got tested
cold) to continuous-discharge-truncation (ported unchanged from
`experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/simulate_batteries_continuous_discharge_truncated.py`):
solve ONE full discharge per battery from `Initial_SOC=0.99`, then
slice that single trace at all 12 target SOC points. This matters
because experiment 26 already proved that training on REST-initialized
low-SOC data collapses sim-to-real accuracy to ~50-65% against every
real dataset here (EMPA/exp07/CALCE) -- none of which are ever
rested-then-resumed, only continuously-discharging cells that simply
arrived at low SOC. Simply widening the old generator's SOC floor
would have reintroduced that exact bug; this avoids it.

Kept unchanged from attempt 03: NMC diffusivity/10 fix, LFP OCP rate
=-3 fix, SOH-scaling, the validated `NEG_CAPACITY_FACTOR=0.45`
LFP-only negative-electrode rebalance, `IDAKLUSolver`.

**Also ported** (per the user's request to check Laswer5's
`optimize-runtime` branch for speedups -- its Polars/Python>=3.12
rewrite isn't usable here, confirmed Python 3.9.6/pybamm 25.6.0, but
one piece is fully portable): capping `OMP_NUM_THREADS`/
`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`VECLIB_MAXIMUM_THREADS` to 1
before numpy/pybamm import, so `ProcessPoolExecutor` workers don't
each spawn competing BLAS threads on top of the worker parallelism.
`SIM_WORKERS` now defaults to all cores (was capped at 4). Measured:
**498 batteries in 22.1s with 8 workers** -- actually faster than the
prior rest-based generator's own 25-28s/4-workers benchmark, despite
every job here solving a full discharge instead of a mix of full and
partial ones.

**Evaluation**: one `RandomForestClassifier`/`XGBClassifier` pair
trained once on the full pooled synthetic set (5,976 batteries,
`Initial_SOC` 0.05-1.0), evaluated separately against each real EMPA
`Initial_SOC` bucket -- same hyperparameters as every prior pooled
baseline (including experiment 22's 97.89% RF result) for a clean
comparison. No ratio-to-median transform, no per-bucket retraining --
isolates the one variable under test.

## Results

| Initial_SOC | n | RF balanced | RF LFP rec | RF NMC rec | XGB balanced | XGB LFP rec | XGB NMC rec |
|---|---|---|---|---|---|---|---|
| 1.0 | 12749 | 98.35% | 1.000 | 0.967 | 98.56% | 0.993 | 0.978 |
| 0.9 | 12750 | 96.88% | 0.970 | 0.968 | 97.55% | 0.970 | 0.981 |
| 0.8 | 12750 | 97.91% | 0.992 | 0.966 | 97.83% | 0.977 | 0.980 |
| 0.7 | 12750 | 97.19% | 0.977 | 0.967 | 97.83% | 0.977 | 0.980 |
| 0.6 | 12750 | 98.40% | 1.000 | 0.968 | 98.23% | 0.984 | 0.980 |
| 0.5 | 12750 | 98.23% | 1.000 | 0.965 | 98.48% | 0.992 | 0.978 |
| 0.4 | 12750 | 98.30% | 1.000 | 0.966 | 98.12% | 0.984 | 0.978 |
| 0.3 | 12750 | 98.29% | 1.000 | 0.966 | 98.53% | 0.992 | 0.979 |
| 0.2 | 12750 | 98.24% | 1.000 | 0.965 | 98.89% | 1.000 | 0.978 |
| 0.15 | 12750 | 98.31% | 1.000 | 0.966 | 98.97% | 1.000 | 0.979 |
| 0.1 | 12750 | 98.41% | 1.000 | 0.968 | 99.08% | 1.000 | 0.982 |
| 0.05 | 10824 | 98.47% | 1.000 | 0.969 | 98.53% | 0.989 | 0.981 |

**Random Forest's collapse is completely fixed.** 96.88-98.47%
balanced accuracy at every bucket, essentially matching XGBoost
(97.19-99.08%) throughout -- confirms the diagnosis: it was the small
per-bucket training sets, not a data or configuration problem with RF
itself. At the pooled scale (5,976 rows), `max_depth=10` is no longer
excessive.

**The accuracy-vs-SOC curve is now flat.** Unlike every per-SOC-
specialized attempt (01-05), which all showed some degradation at
low SOC (attempt 05's best result still dropped from ~99.9% at SOC>=0.3
to 95.17% at SOC=0.1 and 79.84% at SOC=0.05), this pooled model holds
96.9%+ (RF) / 97.2%+ (XGB) all the way down to SOC=0.05 -- no cliff
at all. Comparing directly to attempt 05's own XGBoost numbers: this
experiment trails slightly at the highest SOC points (97-99% here vs.
~99.9% there) but is dramatically better at the low end (98.53% here
vs. 79.84% there at SOC=0.05; 99.08% here vs. 95.17% there at
SOC=0.1). Net: a less sharply-tuned peak, but a far more robust model
across the full range -- directly matching the actual goal (one model,
unknown test-time SOC).

## Interpretation

Confirms both hypotheses at once: per-SOC specialization was
unnecessary complexity that (a) caused Random Forest's collapse via
undersized per-bucket training sets, and (b) wasn't even solving the
right problem -- a single model trained on genuinely full-SOC-range,
physically-correct (continuous-discharge-truncated) synthetic data
generalizes better across the SOC sweep than any of the per-bucket
specialists did, with no engineering workaround (ratio-to-median,
fixed median-reference bins, SOC-matched training) required.

## Files

- `data_generation/simulate_batteries.py` -- continuous-discharge-truncated
  generator with the thread-cap fix; writes to
  `data/optimize_empa_low_soc_06_synthetic/`.
- `data_generation/soc_truncation.py` -- unchanged local copy (from
  experiment 26).
- `pipeline/feature_engineering.py` -- unchanged local copy.
- `run_evaluation.py` -- pooled training + per-bucket evaluation.
- `make_plot.py` -- SOC-sweep accuracy plot.
- `outputs/sweep_results.json`, `outputs/plots/soc_sweep_accuracy_empa.png`.
- `outputs/models/` -- both fitted models (`random_forest.joblib`,
  `xgboost.joblib`) + shared preprocessing (`scaler.joblib`,
  `imputer.joblib`, `label_encoder.joblib`, `clip_bounds.json`,
  `feature_names.json`) -- one shared pipeline now, not one per bucket.

## Next steps

Not pursued here: re-adding ratio-to-median on top of this pooled,
full-range setup (unclear it's still needed, since the flat accuracy
curve suggests the model already generalizes across magnitude
differences without it); increasing `VARIATIONS_PER_SIZE` beyond 83
for even more training data (kept at the proven-working scale here to
isolate this experiment's one actual change).
