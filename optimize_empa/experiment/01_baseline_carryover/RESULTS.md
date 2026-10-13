# optimize_empa_low_soc, attempt 01: baseline (carried over from the colleague-branch merge)

## What this is

Not new work -- this is the current best-known, bug-fixed, full-scale-
verified EMPA SOC-sweep result, re-homed into this campaign's own
folder/data structure as the starting point for low-SOC optimization.

## Where this pipeline came from

On this same branch (`optimize-empa-2-merge`), three colleague branches
(`hilda_empa_op`/`main`, `ella_experimental_data`, `optimize-runtime`)
had each independently modified `optimize_empa_2/`'s SOC-sweep pipeline
after it was found to collapse to ~50% balanced accuracy across the
sweep. Comparing them found:

- **Hilda's `main`**: one Random Forest + XGBoost model trained once on
  the full synthetic set, tested per bucket; bin selection via the
  shared `feature_engineering.py`'s own internal filter, which pools
  synthetic+real together by chemistry (a test-set-leakage-into-
  feature-selection issue); `MAX_CYCLES_PER_CELL=600`,
  `SOH_MIN=0.85`, `C_RATE` 0.05-1.0.
- **Ella's branch**: a fresh model per SOC bucket, a hybrid feature
  representation (raw voltage resampled onto a 0-1 capacity-fraction
  grid for SOC<=0.3, standard dV/dQ bins above), `class_weight='balanced'`
  on the Random Forest, no depth-of-discharge sanity filter on real
  cycles.
- **Laswer5's `optimize-runtime`**: a from-scratch Polars/Parquet/`uv`
  rewrite (Python >=3.12, not runnable in this repo's Python 3.9
  environment) using `pybamm.IDAKLUSolver` with a single precomputed
  `t_interp` pass instead of the original two-pass splice, parallelized
  across worker processes, and bin coverage computed from training
  (synthetic) data only.

## What was merged into this pipeline

1. **Training-only bin coverage selection** (ported from Laswer5's
   idea, implemented in plain pandas rather than adopting his
   polars/`uv` stack): `create_features_by_voltage_bins()` is called
   with `min_chemistry_coverage=0.0` (disabling its internal pooled
   filter), then bins are independently re-checked for >=20% coverage
   using *only* the synthetic/training subset -- the real/test side
   never influences which bins get trained on.
2. **Per-SOC specialized models** (Ella's structural idea, without her
   untested capacity-fraction feature): a fresh Random Forest + XGBoost
   pair per real `Initial_SOC` bucket, with synthetic training rows
   narrowed to within +/-0.1 of that bucket's target SOC wherever the
   generator's own range (`Initial_SOC` 0.5-1.0) makes that possible.
3. **`class_weight='balanced'`** on the Random Forest (Ella's branch).
4. **A fast synthetic-data generator**: `pybamm.IDAKLUSolver` with one
   precomputed `t_interp` pass instead of the original two-pass splice,
   parallelized across up to 4 worker processes. Measured in this
   environment before adopting: ~2x faster per solve; verified the
   solver still terminates exactly on the voltage-cutoff event even
   when the time-horizon estimate undershoots the real duration.
   Full 498-battery run: **~25-28 seconds** (the original serial
   two-pass script would realistically take 15-30+ minutes).
5. **Chunked real-data feature extraction**: at the full
   `MAX_CYCLES_PER_CELL=600` scale, real raw data is ~12M rows / 1.7GB
   -- loading that as one DataFrame through the shared
   `feature_engineering.py`'s sort/vectorized-diff/pivot pipeline
   risked repeating an out-of-memory crash seen earlier in this project
   on a different pipeline. Processed in ~1M-row chunks of real
   `Variation_ID`s instead, each through a small temp CSV and the same
   shared feature-extraction call, keeping only the small aggregated
   (one row per battery) result before the next chunk is built.
   Verified safe with active RSS monitoring through this run: peaked at
   **2.6GB**, no danger-threshold trips.

## A real bug found and fixed along the way

The first version of the merged evaluator re-derived `Battery_ID` via
`raw_df.groupby(GROUPBY_COLS).ngroup()` (default `sort=True`) to map
`DataKind`/`Initial_SOC` metadata back onto
`create_features_by_voltage_bins()`'s output -- but that function
assigns its own `Battery_ID` internally via `ngroup(sort=False)`. Two
different numbering schemes for the same groups meant `DataKind` was
silently mapped onto the wrong battery rows. This **inflated an
earlier, smaller-scale smoke test to a misleadingly perfect 97.5-99.6%
balanced accuracy across every SOC bucket** -- that number is wrong and
should be disregarded. The chunked design here sidesteps the bug
entirely (each extraction call gets homogeneous input, so `DataKind` is
set directly, no re-derivation needed). Verified the fix by reproducing
the synthetic-only coverage check in isolation: deterministically
reports the same 14 surviving bins seen in this run.

## Results (full scale: 498 synthetic batteries, 151,074 real cycles / 12.1M rows)

| Initial_SOC | n (real) | n_train (synth) | SOC-matched? | RF balanced | RF LFP recall | XGB balanced | XGB LFP recall |
|---|---|---|---|---|---|---|---|
| 1.0 | 12750 | 86 | yes | 49.87% | 0.000 | 99.89% | 1.000 |
| 0.9 | 12750 | 194 | yes | 49.87% | 0.000 | 99.89% | 1.000 |
| 0.8 | 12750 | 212 | yes | 49.87% | 0.000 | 99.87% | 1.000 |
| 0.7 | 12750 | 196 | yes | 49.87% | 0.000 | 95.72% | 0.917 |
| 0.6 | 12750 | 200 | yes | 49.87% | 0.000 | 99.89% | 1.000 |
| 0.5 | 12750 | 110 | yes | 49.87% | 0.000 | 99.91% | 1.000 |
| 0.4 | 12750 | 498 | no (extrapolation) | 49.93% | 0.000 | 99.95% | 1.000 |
| 0.3 | 12750 | 498 | no (extrapolation) | 49.99% | 0.000 | 99.99% | 1.000 |
| 0.2 | 12750 | 498 | no (extrapolation) | 49.99% | 0.000 | 98.43% | 0.969 |
| 0.15 | 12750 | 498 | no (extrapolation) | 49.99% | 0.000 | **49.99%** | **0.000** |
| 0.1 | 12750 | 498 | no (extrapolation) | 49.89% | 0.000 | **50.00%** | **0.000** |
| 0.05 | 10824 | 498 | no (extrapolation) | 50.00% | 0.000 | **50.00%** | **0.000** |

**Random Forest predicts the majority class (NMC) almost everywhere**,
`LFP_recall=0.000` in every single bucket, despite `class_weight='balanced'`.
**XGBoost holds up well through Initial_SOC 1.0-0.2** (95.7-99.9%
balanced accuracy) **then also collapses to chance at 0.15 and below.**

## Root cause, as far as it's been diagnosed

A controlled 2x2 comparison (bin set: training-only vs. pooled-coverage
x training data: SOC-matched vs. full-synthetic-set, with and without
`class_weight`) showed **every combination collapses identically** --
none of this pipeline's own methodology choices explain it.

Comparing median dV/dQ per bin by chemistry at Initial_SOC=1.0:

| Bin | Synthetic LFP | Synthetic NMC | Real LFP | Real NMC |
|---|---|---|---|---|
| 3.1-3.0V | -0.44 | -2.15 | **-1.87** | -3.41 |
| 3.0-2.9V | -0.50 | -2.96 | **-2.65** | -4.39 |
| 2.9-2.8V | -1.24 | -4.94 | **-3.50** | -5.92 |

**Real LFP's dV/dQ magnitude in the range it actually populates
(2.7-3.3V) sits much closer to synthetic NMC's calibration than to
synthetic LFP's own calibration.** A classifier trained purely on
synthetic LFP's (much flatter) profile in this range has little basis
to recognize real LFP -- it looks like NMC. This is a genuine
calibration gap between the synthetic LFP OCP model and this real
EMPA dataset, not a bug in this merge: directly compared this
pipeline's fast (IDAKLU) synthetic generator against the original slow
(two-pass) generator and found near-identical LFP calibration between
them (e.g. 3.1-3.0V: -0.44 fast vs. -0.46 original) -- the mismatch
predates and is independent of anything built in this merge.

**Unresolved**: colleagues' own committed `optimize_empa_2/sweep_results.json`
(generated by what appears to be the same settings -- committed in the
same commit that tightened them, and reporting real test-set sizes
that exactly match this run's, bucket for bucket) reports Random
Forest at a flat ~90% balanced accuracy, not this collapse. Every
checkable difference (bin selection, SOC-matching, `class_weight`,
real data source, synthetic generator calibration) has been ruled out.
The most likely remaining explanation is that colleagues' local
`data/raw/advanced_synthetic_battery_data.csv` -- entirely gitignored,
never reproducible from git history alone -- predates a settings or
physics change their committed code reflects, but this hasn't been
confirmable from here. Not pursued further per explicit direction; this
campaign proceeds from the honest, bug-fixed, full-scale result above.

## Files

- `run_evaluation.py` -- self-contained per-SOC evaluation (EMPA only). The script you run.
- `data_generation/` -- `simulate_batteries.py` (fast parallel generator) and
  `build_real_empa.py`. Not invoked by `run_evaluation.py` -- kept so the
  input data is reproducible from local files alone.
- `pipeline/feature_engineering.py` -- local copy, imported by `run_evaluation.py`.
- `outputs/sweep_results.json` -- this attempt's results.

## Next steps

XGBoost's collapse specifically between SOC=0.2 and SOC=0.15 lines up
with where real buckets stop having any SOC-matched synthetic training
data at all (`SYNTHETIC_SOC_FLOOR=0.5`) is a red herring -- *every*
bucket below 0.5 already uses the same full-synthetic fallback, yet
0.2 still works (98.4%) while 0.15 doesn't (50.0%). The real driver is
more likely the LFP calibration gap documented above compounding with
real LFP sample sparsity at very low SOC. The calibration gap itself is
the more promising lever: `optimize_empa/experiment/01_ratio_to_median_baseline/`
(a separate, earlier campaign, same underlying classification problem)
found a **negative-electrode capacity rebalance** (scaling LFP's
negative-electrode max/initial concentration by a factor independent of
the positive electrode) shifts the synthetic LFP dV/dQ dive to better
match real EMPA data, and validated it end-to-end (not just a narrow
diagnostic) -- see that folder's RESULTS.md for the full derivation.
Attempt 02 applies the same lever here.
