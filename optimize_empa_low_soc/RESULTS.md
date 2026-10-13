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

## Background

Full derivation and debugging history (the colleague-branch comparison,
a `Battery_ID`/`DataKind` mapping bug found and fixed, a root-caused
LFP OCP calibration mismatch, and an inconclusive investigation into
why colleagues' own reported numbers didn't show this collapse) lives
in this session's conversation history and in `experiment/01_baseline_carryover/RESULTS.md`.
Not repeated in full here -- this campaign builds on that conclusion
rather than re-deriving it.
