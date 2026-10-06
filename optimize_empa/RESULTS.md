# optimize_empa: campaign index

## Objective

Per a new directive: focus exclusively on optimizing balanced accuracy
for the EMPA dataset. exp07, SNL, and CALCE are out of scope for this
campaign -- any finding here is judged ONLY against EMPA's own real
discharge data.

This is a new top-level folder (not nested under `experiments/`, and not
numbered in that scheme) since it's an ongoing campaign that will likely
span multiple optimization attempts, not a single one-off experiment.

## Data sources

Both real and synthetic data live in this campaign's own
`data/optimize_empa_NN_.../` namespace (byte-for-byte copies of the
upstream data, not regenerated), so each attempt's folder is runnable
independently of any other experiment's data path.

- **Real EMPA**: `data/optimize_empa_01_real_empa/raw/real_empa_soc_sweep_raw.csv`
  -- a byte-for-byte copy of `data/26_real_empa_soc_sweep_soh_range/`
  (the actual downloaded EMPA RO-Crate data: 199 physical cells, 35,629
  discharge cycles pooled across all 12 Initial_SOC points). Content
  unchanged, only the path is campaign-local; originally built by
  `experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/build_real_empa_soc_sweep_soh_range.py`
  (local copy: `experiment/01_ratio_to_median_baseline/data_generation/build_real_empa_soc_sweep_soh_range.py`).
- **Synthetic**: each attempt's own `data/optimize_empa_NN_.../` folder
  (separate from the old `experiments/32_` lineage) -- see each attempt's
  own RESULTS.md for what it contains.

## Running log of attempts

| # | Folder | Idea | Best balanced accuracy (EMPA) | Status |
|---|---|---|---|---|
| 01 | `experiment/01_ratio_to_median_baseline/` | Carry over the best-known combination from experiments/32_fixed_voltage_cutoff_soc_sweep/ (negative-electrode-balance-fixed LFP + ratio-to-median scale-invariant features + per-chemistry coverage fix) as this campaign's starting baseline | 92.8-99.9% from Initial_SOC 1.0 down to 0.4; collapses to chance (38-51%) below 0.3 | Baseline, verified (re-run from scratch in the new location, reproduces experiment 32's numbers exactly) -- see its own RESULTS.md |

## Background: how attempt 01's baseline was found

Full history lives in `experiments/32_fixed_voltage_cutoff_soc_sweep/RESULTS.md`
(four stages: a forced-voltage-cutoff dead end, per-SOC-point
specialization, a negative-electrode-capacity-balance physics fix, and
scale-invariant ratio-to-median features) and in project memory
([[exp29_lfp_ocp_tail_ceiling]], [[exp29_negative_electrode_diffusivity_fix]],
[[exp27_snl_generalization_gap]]). Not repeated here -- this campaign
builds on that conclusion rather than re-deriving it.
