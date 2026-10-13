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

## Current best, and a correction to attempt 04's "next steps"

Attempt 04 proposed a per-bucket hybrid (attempt 03's Random Forest for
Initial_SOC 1.0-0.8, attempt 04's XGBoost below) on the premise that
attempt 03's RF "won" at the top of the sweep. Checking that premise
properly -- comparing **all four** model/feature combinations
(attempt 03 RF/XGB, attempt 04 RF/XGB) head-to-head at every SOC point,
not just RF-under-ratio-to-median against RF-under-raw-features --
found that premise was wrong:

| SOC | A03 RF | A03 XGB | A04 RF | A04 XGB | Best |
|---|---|---|---|---|---|
| 1.0 | 99.87 | 99.87 | 50.00 | **99.95** | A04-XGB |
| 0.9 | 99.87 | 99.87 | 50.00 | **99.94** | A04-XGB |
| 0.8 | 63.93 | 99.87 | 50.00 | **99.98** | A04-XGB |
| 0.7-0.5 | ~50 | ~99.9 | ~50 | ~99.9-100 | A04-XGB (narrowly) |
| 0.4 | ~50 | 99.91 | ~50 | 99.89 | A03-XGB (narrowly) |
| 0.3-0.2 | ~50 | 99.99 | ~50 | **100.00** | A04-XGB |
| 0.15 | ~50 | **99.99** | ~50 | 99.92 | A03-XGB (narrowly) |
| 0.1 | 50.00 | 49.99 | 50.00 | 48.98 | all chance |
| 0.05 | 50.00 | 50.00 | 50.00 | **81.81** | A04-XGB |

**XGBoost wins or ties at every single SOC point, regardless of
feature set.** Random Forest never actually wins anywhere -- attempt
03's RF successes (1.0/0.9/0.8) are matched or exceeded by attempt
04's XGBoost. The differences between attempt 03's and attempt 04's
XGBoost are noise-level (<0.1pp) everywhere except Initial_SOC=0.05,
where attempt 04 is dramatically better.

**Conclusion: attempt 04's ratio-to-median + XGBoost pipeline, alone,
is already this campaign's best unified result across the sweep** --
no hybrid needed. Random Forest can be dropped from future attempts;
it's dominated by XGBoost in every configuration tested so far.

**Remaining open question, not yet resolved**: Initial_SOC=0.1 stays
at chance for XGBoost in both attempts, while the more extreme
Initial_SOC=0.05 works (81.81%) -- counter-intuitive, since 0.1 is the
easier extrapolation case (closer to the synthetic generator's own
range) and uses literally the same training data (both buckets fall
back to the full 498-battery synthetic set). A quick follow-up check
found the real data's own LFP-vs-NMC separation pattern is
qualitatively similar at both SOC points (LFP's ratio-to-median
magnitude exceeds NMC's at the same bins in both cases) -- so this
isn't an "0.1 has no signal" problem, it's a decision-boundary
placement issue relative to where each bucket's real data happens to
sit. Not root-caused further; flagged for future work rather than
chased down an open-ended rabbit hole.

## Background

Full derivation and debugging history (the colleague-branch comparison,
a `Battery_ID`/`DataKind` mapping bug found and fixed, a root-caused
LFP OCP calibration mismatch, and an inconclusive investigation into
why colleagues' own reported numbers didn't show this collapse) lives
in this session's conversation history and in `experiment/01_baseline_carryover/RESULTS.md`.
Not repeated in full here -- this campaign builds on that conclusion
rather than re-deriving it.
