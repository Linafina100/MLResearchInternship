# optimize_empa, attempt 01: baseline (carried over from experiment 32)

## What this is

Not new work -- this is the current best-known EMPA result, re-homed
into this campaign's own folder/data structure as the starting point
for further optimization. Full derivation history (why this specific
combination of fixes, including two dead ends ruled out along the way)
lives in `experiments/32_fixed_voltage_cutoff_soc_sweep/RESULTS.md`
(Stages 1-4) -- summarized here, not repeated in full.

## The combination

1. **Per-SOC-point specialized models**: one Random Forest trained per
   Initial_SOC point (not one pooled model across all 12) -- pooling
   measurably blurs sensitivity to bins a heavily-truncated low-SOC
   example has no data in.
2. **Negative-electrode capacity rebalance** (LFP only, factor=0.7):
   fixes a genuine physics mismatch -- real EMPA LFP's dV/dQ dive
   happens by ~2.65V, but the unmodified synthetic calibration's
   equivalent steepness only appeared near 2.3-2.5V, a region real EMPA
   cells never reach.
3. **Ratio-to-median scale-invariant features**: each bin's dV/dQ
   divided by that sample's own median across its own observed bins,
   instead of absolute dV/dQ (which encodes physical cell size, not
   usable at Stena's production line where only a partial window at
   unknown SOC/SOH is ever observed).
4. **Per-chemistry coverage filtering**: the shared
   `feature_engineering.py` utility's coverage filter checks POOLED
   (synthetic+real, both chemistries) coverage, which can let a bin
   survive even when one chemistry's synthetic side has ZERO coverage
   there -- corrupting that feature for the entire affected class. Fixed
   with an additional per-(DataKind,Chemistry) coverage re-check.

## Data (copied to this campaign's own namespace, not regenerated)

- Synthetic: `data/optimize_empa_01_synthetic_baseline/` -- a byte-for-
  byte copy of `data/32_synthetic_lfp_balance_fix_combined/` (249 LFP +
  249 NMC base battery configs; LFP has the negative-electrode-balance
  fix, NMC is the unmodified baseline).
- Real: `data/26_real_empa_soc_sweep_soh_range/` (unchanged, 199 cells,
  35,629 cycles).

## Results

| Initial_SOC | Balanced accuracy |
|---|---|
| 1.0 | 92.84% |
| 0.9 | 99.95% |
| 0.8 | 99.87% |
| 0.7 | 99.92% |
| 0.6 | 99.64% |
| 0.5 | 99.09% |
| 0.4 | 99.06% |
| 0.3 | 50.00% |
| 0.2 | 38.09% |
| 0.15 | 48.59% |
| 0.1 | 50.60% |
| 0.05 | n/a (too little synthetic data survives the coverage filter) |

![EMPA balanced accuracy, attempt 01](plots/per_soc_accuracy.png)

Re-run from scratch in this new, self-contained location (new data
path, new script) rather than copied from experiment 32 -- confirmed to
reproduce the exact same numbers, validating the new wiring.

**Reliable range: Initial_SOC 1.0 down to 0.4** (92.8-99.9% balanced
accuracy). **Below Initial_SOC 0.3, accuracy collapses to chance**
(38-51%) -- a real, physical limit (too little voltage range survives
in a narrow truncated window), not something this feature engineering
approach fixes on its own. This low-SOC range is the natural target for
the next optimization attempt.

## Files

- `run_evaluation.py` -- self-contained per-SOC evaluation (EMPA only). The script you run.
- `make_plot.py` -- reads `outputs/per_soc_results.json`, draws the plot. Run after.
- `pipeline/` -- local copies of shared pipeline code `run_evaluation.py` depends on:
  `feature_engineering.py` (imported directly) and `ml_pipeline.py` (copied for
  self-containment, not currently invoked).
- `data_generation/` -- local copies of the scripts that produced this
  experiment's synthetic + real CSVs (`simulate_batteries_continuous_discharge_truncated.py`,
  `simulate_lfp_negative_electrode_balance.py`, `combine_lfp_fix_with_baseline_nmc.py`,
  `soc_truncation.py`, `build_real_empa_soc_sweep_soh_range.py`). None of these are
  invoked by `run_evaluation.py` -- kept only so the input data is reproducible
  from local files alone, without depending on the old `experiments/32_` lineage.
- `outputs/` -- `per_soc_results.json`, `features/` (per-SOC extracted feature
  CSVs), `plots/per_soc_accuracy.png`.
