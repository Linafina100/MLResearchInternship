# optimize_empa_low_soc: campaign index

## Objective

Starting from the bug-fixed, full-scale-verified EMPA SOC-sweep pipeline
built while merging three colleagues' branches (`optimize_empa_2/` on
this same branch, `optimize-empa-2-merge`), push balanced accuracy as
high as possible at as low an Initial_SOC as possible. Deliberately
kept separate from `optimize_empa_2/` (the colleagues' shared pipeline
-- not ours to restructure) and from `experiments/32_fixed_voltage_cutoff_soc_sweep/`
(a different, already-concluded campaign on `main`).

## Data sources

Each attempt's own `data/optimize_empa_low_soc_NN_.../` namespace
(synthetic and real both), self-contained and not shared with
`optimize_empa_2/`'s `data/raw/` paths, so this campaign's data can be
regenerated or modified without touching the colleagues' pipeline.

## Running log of attempts

| # | Folder | Idea | Best result | Status |
|---|---|---|---|---|
| 01 | `experiment/01_baseline_carryover/` | Carry over the bug-fixed, full-scale-verified `optimize_empa_2` pipeline unchanged, as this campaign's starting point | XGBoost: 95.7-99.9% from SOC 1.0 down to 0.2, collapses to chance at 0.15 and below. Random Forest: chance-level (majority-class prediction) at every SOC point | Baseline, verified (re-run from scratch in this new, self-contained location; reproduces the merge work's full-scale numbers exactly -- see its own RESULTS.md) |
| 02 | `experiment/02_negative_electrode_balance/` | LFP negative-electrode capacity rebalance (factor=0.7, carried over from `optimize_empa/experiment/01_...`), targeting the LFP OCP calibration gap root-caused in attempt 01 | XGBoost: working range extended a full SOC step lower, now 95.7-99.9% down to **0.15** (was 0.2). Random Forest: recovered at SOC 1.0/0.9 (99.87%, was chance), still chance-level 0.8 and below. Both models still chance below 0.1 | Verified at full scale; a real, partial improvement -- see its own RESULTS.md for the full before/after table and the sanity-check diagnostic that preceded the full run |
| 03 | `experiment/03_tuned_negative_electrode_factor/` | Swept `NEG_CAPACITY_FACTOR` (0.4-0.8) against this pipeline's own calibration gap instead of assuming attempt 02's borrowed 0.7 was optimal; found 0.45-0.47 fits real LFP noticeably better | Random Forest partially recovers at SOC=0.8 (LFP recall 0.000 -> 0.281). XGBoost essentially unchanged (already saturated 1.0-0.15 with the looser factor) -- no further SOC-range extension. Diminishing returns from this lever alone | Verified at full scale -- see its own RESULTS.md; recommends a different/additional lever next (ratio-to-median scale-invariant features) |
| 04 | `experiment/04_ratio_to_median_features/` | Ratio-to-median scale-invariant features (each bin's dV/dQ divided by that row's own median), layered on attempt 03's synthetic data -- a feature-representation fix rather than a physics fix | Mixed: Random Forest regresses to chance everywhere it was working in attempt 03 (1.0/0.9/0.8). XGBoost breaks through at the lowest SOC point in the whole campaign: Initial_SOC=0.05 goes from chance to **81.81%** balanced accuracy (LFP recall 0.000 -> 1.000) | Verified at full scale -- see its own RESULTS.md; neither attempt 03 nor 04 alone is best across the whole sweep, suggesting a per-bucket hybrid next |

## Background

Full derivation and debugging history (the colleague-branch comparison,
a `Battery_ID`/`DataKind` mapping bug found and fixed, a root-caused
LFP OCP calibration mismatch, and an inconclusive investigation into
why colleagues' own reported numbers didn't show this collapse) lives
in this session's conversation history and in `experiment/01_baseline_carryover/RESULTS.md`.
Not repeated in full here -- this campaign builds on that conclusion
rather than re-deriving it.
