# Experiment 16: low-voltage-only sim-to-real via a lowered simulation cutoff

## Motivation

The real-world use case (battery recycling) needs the classifier to work
on already-mostly-depleted cells -- i.e. only the low-voltage tail of the
discharge curve. Real LFP lab data is cut off at exactly 2.5V; real NMC
lab data's observed floor is also ~2.5V (see
`../07_real_lfp_nmc_test/parse_real_lfp.py` / `parse_real_nmc.py`). Every
sim-to-real test built so far (`../07_real_lfp_nmc_test/02_sim_to_real/`)
trains on the *entire* discharge curve, which includes voltage regions the
real use case never sees, and sits right where the known PyBaMM
solver-termination artifact lives (event-triggered `"Discharge until X V"`
produces one abnormally large final step; for a minority of batteries,
more than one oversized step near the cutoff -- see
[[exp07-sim-to-real-failure]]).

**Proposed fix ("the cutoff hack")**: artificially lower the *simulated*
cutoff well below the real 2.5-3.0V target zone, so the whole artifact
(including any multi-step cascade) lands safely out of range, then
restrict the feature set to only the real-world-relevant low-voltage
bins.

## Phase 1: diagnostic -- is the mechanism actually safe?

`diagnose_lower_cutoff.py` simulates a small batch (LFP + all 3 NMC
parameter sets, 3 SOH values: 0.50/0.675/0.85, one capacity tier, full
initial SOC) at three candidate cutoffs (1.5V, 1.0V, 0.5V), and for each
run checks: (a) whether PyBaMM solves without error, and (b) whether any
raw transition whose *ending* voltage lands in the 2.5-3.0V target zone
has an outlier |dV/dQ| (threshold 50 -- comfortably above the largest
genuine values seen anywhere in this project's real or artifact-free
synthetic data, which never exceed ~10).

This directly tests the mechanism, rather than assuming it from analogy:
root `feature_engineering.py` bins every transition using only its ending
voltage (`v_val = group['Voltage [V]'].iloc[idx]`), single bin, never
split across bins -- so if the artifact's ending voltage stays below 2.5V,
it cannot contaminate the target zone's bins by construction. What wasn't
known in advance was whether a much lower cutoff produces the same,
smaller, or larger jumps than the 1.8V/2.3V cutoffs used elsewhere, or
whether PyBaMM even converges reliably that low.

**Result: the target zone is clean at every candidate cutoff tested.**

| v_min | Solves failed | Outlier transitions in 2.5-3.0V zone | Max \|dV/dQ\| in zone |
|---|---|---|---|
| 1.5V | 2/12 (unrelated, see below) | 0 | 3.90 |
| 1.0V | 2/12 (unrelated, see below) | 0 | 3.90 |
| 0.5V | 2/12 (unrelated, see below) | 0 | 3.90 |

Zero outlier transitions landed in the 2.5-3.0V zone at any tested
cutoff, and the largest genuine dV/dQ magnitude observed there (3.90) is
well within the range real data itself shows. Every failed cutoff
level shows the exact same 2 failures (NMC Chen2020 and OKane2022, both
at SOH=0.50): `Q_Li=3.0443 Ah is outside the range of possible values
[0.0000, 2.9120]`, identical across all three v_min values -- this is a
parameter-construction issue from this diagnostic's fixed `initial_soc=1.0`
combined with a low SOH scaling factor, unrelated to the voltage cutoff
(it doesn't get worse or better as v_min changes). Root
`simulate_batteries.py` draws `initial_soc` randomly from 0.5-1.0 (not
fixed at 1.0), and already has a built-in solve-failure log-and-skip
mechanism (`solve_with_cutoff`) for exactly this kind of rare parameter
mismatch, so this is not expected to meaningfully affect Phase 2.

**Chosen cutoff: 1.5V for both chemistries.** All three candidates passed
equally cleanly, so the least aggressive one was chosen -- no reason to
push further from the existing 1.8V/2.3V convention than necessary. This
gives a full 1.0V margin between the simulated cutoff and the 2.5V target
floor, comfortably above the largest historically observed contamination
reach (~0.6V, from experiment 07's multi-jump investigation at the
existing, higher cutoffs).

Raw diagnostic output: `diagnostic_results.csv`.

## Phase 2: full sim-to-real test restricted to the 2.5-3.0V zone

A full synthetic batch was simulated at the chosen 1.5V cutoff (both
chemistries) via unmodified root `simulate_batteries.py`
(`LFP_LOWER_CUTOFF=1.5 NMC_LOWER_CUTOFF=1.5 RUN_LABEL=low_voltage_v1.5`,
same random SOC/SOH/C-rate/parameter-set diversity as every other
experiment). **13 of 498 solve attempts failed (2.6%)** -- the same
`Q_Li ... outside the range of possible values` pattern Phase 1 found,
unrelated to the cutoff, plus one unrelated `SolverError` -- leaving 485
usable synthetic batteries, a healthy sample comparable to other
experiments' scale.

`evaluate_low_voltage_sim_to_real.py` combined this synthetic data with
experiment 07's real LFP/NMC data, extracted features once with root
`feature_engineering.py` (`exclude_final_transition=True`), and restricted
training/testing to only the 5 bins inside 2.5-3.0V
(`dV_dQ_V_3.0_2.9` ... `dV_dQ_V_2.6_2.5`).

**Result: 24.33% accuracy, both models** -- essentially identical to
experiment 07's full-curve baseline (24.84%) and the naive
always-predict-LFP floor. NMC recall was 0.00 (worse than the full-curve
baseline's 0.01) -- the model still collapses to predicting LFP almost
universally.

**The cutoff-hack mechanism itself worked exactly as designed**: no
outlier/artifact dV/dQ values were found in the target zone (consistent
with Phase 1's diagnostic). But a different, larger problem dominates the
result, found by comparing real vs. synthetic mean dV/dQ and per-chemistry
*data coverage* (fraction of batteries with a genuine, non-imputed value)
in each target-zone bin:

| Bin | Real LFP mean | Synth LFP mean | Real LFP coverage | Synth LFP coverage | Real NMC mean | Synth NMC mean | Real NMC coverage | Synth NMC coverage |
|---|---|---|---|---|---|---|---|---|
| 3.0-2.9 | -0.31 | -1.91 | 99.5% | 55.8% | -3.36 | -15.49 | 99.4% | 0.9% |
| 2.9-2.8 | -0.71 | -2.97 | 99.5% | 44.2% | -2.73 | -2.23 | 100% | 0.4% |
| 2.8-2.7 | -1.18 | -3.77 | 99.5% | 30.5% | -3.30 | NaN | 100% | 0.0% |
| 2.7-2.6 | -1.33 | -5.03 | 99.5% | 18.9% | -5.24 | -1.58 | 100% | 0.4% |
| 2.6-2.5 | -2.86 | -5.78 | 99.5% | 13.3% | -7.73 | NaN | 100% | 0.0% |

Two distinct problems, both pre-existing and neither fixed by the cutoff
hack:

1. **LFP: a systematic 2-6x magnitude mismatch**, growing worse at lower
   voltage (synthetic dV/dQ 2-6x larger than real across all 5 bins) --
   the same character of synthetic-vs-real mismatch experiment 07 found
   for NMC across the full curve, here showing up for LFP in this
   specific low-voltage window instead. Also, synthetic LFP's own
   coverage collapses steeply toward lower voltage (56% down to 13%) --
   many synthetic LFP batteries pass through this narrow voltage window
   too quickly (few raw samples) to leave data in every bin, even before
   any artifact consideration.
2. **NMC: near-total absence of genuine synthetic data in this zone**,
   artifact-free or not. Real NMC has ~100% coverage in every target-zone
   bin (every real cycle passes smoothly through 2.5-3.0V), but synthetic
   NMC's coverage is 0-0.9% -- essentially no synthetic NMC battery
   leaves a genuine raw sample here at all, two bins entirely empty
   (NaN). This is not the termination artifact (already confirmed absent
   here by Phase 1) -- it's PyBaMM's adaptive solver taking large strides
   through this particular voltage range for NMC regardless of where the
   experiment terminates, an intrinsic sampling-density property of the
   simulated NMC discharge curve shape, not an artifact of the cutoff
   event. This generalizes experiment 07's earlier, narrower finding
   ("no synthetic NMC batteries leave samples inside 2.4-2.9V near the
   *old* cutoff") to: this voltage range is under-sampled by the NMC
   simulation *everywhere*, cutoff-independent.

With synthetic NMC's target-zone bins imputed to a near-constant for 99%+
of NMC training rows (vs. real NMC's genuinely varying values), the model
trained on synthetic data learns a decision rule that does not match how
real data actually looks in this zone -- consistent with the observed
collapse to predicting LFP almost universally.

**Conclusion**: the cutoff hack is a validated, working technique for
isolating the solver-termination artifact from a target voltage zone --
that specific problem is solved. But it doesn't rescue sim-to-real
accuracy here, because a different, more fundamental problem dominates:
PyBaMM's adaptive time-stepping produces too few genuine raw samples for
synthetic NMC in the 2.5-3.0V range to build real signal there, and
synthetic LFP shows a separate systematic magnitude mismatch against real
data in the same window. Fixing either would need forcing denser sampling
specifically through this voltage window (e.g. an explicit finer time/
voltage grid restricted to this range) or investigating the magnitude
mismatch's cause directly -- neither attempted here. Note: experiment 13
found that forcing a finer *global* time grid made the termination
artifact *worse* (right at the cutoff); whether the same holds for
denser sampling *away* from the cutoff, mid-curve, is a distinct,
untested question, not automatically implied by that finding.

Full run outputs: `sim_and_real_raw.csv`, `features/ml_features.csv`,
`features/synthetic_features_target_zone.csv`,
`features/real_features_target_zone.csv` (all gitignored, regenerable via
`diagnose_lower_cutoff.py` then `evaluate_low_voltage_sim_to_real.py`).
