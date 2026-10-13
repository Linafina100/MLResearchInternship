# Experiment 17: does healthier SOH + a much gentler C-rate fix experiment 16's problem?

## Motivation

Experiment 16 (`../16_low_voltage_only_sim_to_real/`) found that even after
solving the solver-termination artifact (by lowering the simulated
cutoff to 1.5V), sim-to-real accuracy in the real-world-relevant
2.5-3.0V zone stayed at the ~24% floor, because synthetic NMC data is
almost entirely *absent* from that zone (0-0.9% coverage vs. real NMC's
~100%) -- PyBaMM's adaptive solver takes large strides through that
voltage range for NMC regardless of cutoff.

This experiment tests a specific hypothesis for fixing that: a much
gentler discharge (C-rate 0.1-0.2, vs. experiment 16's fixed 0.6C) should
reduce IR-drop-driven dynamics and could plausibly cause the adaptive
solver to take *finer* steps throughout the discharge, including through
the target zone. Healthier cells (SOH 0.8-1.0, vs. 0.50-0.85) were tested
alongside this, both because they better match the "not yet very
degraded" end of a recycling stream and because experiment 16's low-SOH
batteries were also the ones hitting an unrelated `Q_Li`-out-of-range
solve failure.

## Method

Same design as experiment 16 in every other respect: `simulate_batteries_high_soh_low_crate.py`
(a copy of root `simulate_batteries.py`, matching the project's
established per-experiment-copy convention used by experiments 06 and
10) with `SOH_RANGE = 0.8-1.0`, `C_RATE = 0.1-0.2` (drawn per battery,
replacing the fixed 0.6C), and the same 1.5V cutoff for both chemistries.
`diagnose_cutoff_at_low_crate.py` re-verified the cutoff's safety margin
first, since C-rate can change solver step dynamics near the cutoff:
**0/36 solves failed, 0 outlier transitions in the target zone at C-rate
0.1/0.15/0.2** -- confirmed safe (and notably, 0 solve failures overall
in the diagnostic, vs. some at experiment 16's lower SOH range).

Full batch: 497/498 batteries simulated successfully (only 1 failure --
a marked improvement over experiment 16's 13/498, consistent with
higher SOH avoiding the earlier `Q_Li`-out-of-range issue). Same
feature-extraction and target-zone-restriction pipeline as experiment 16
(`evaluate_high_soh_low_crate_sim_to_real.py`).

## Result: identical accuracy, 24.33% -- to the decimal

| Model | Experiment 16 (SOH 0.50-0.85, 0.6C) | Experiment 17 (SOH 0.8-1.0, C-rate 0.1-0.2) |
|---|---|---|
| Random Forest | 24.33% | 24.33% |
| XGBoost | 24.33% | 24.33% |

Same failure mode too: NMC recall 0.00, model collapses to predicting
LFP almost universally.

## Why: NMC's coverage gap didn't move; LFP's did, in a way that didn't matter

| Bin | Real LFP | Synth LFP (exp16, 0.6C) | Synth LFP (exp17, 0.1-0.2C) | Real NMC | Synth NMC (exp16) | Synth NMC (exp17) |
|---|---|---|---|---|---|---|
| Coverage 3.0-2.9 | 99.5% | 55.8% | **13.3%** | 99.4% | 0.9% | 0.8% |
| Coverage 2.9-2.8 | 99.5% | 44.2% | **5.6%** | 100% | 0.4% | 0.4% |
| Coverage 2.8-2.7 | 99.5% | 30.5% | **4.0%** | 100% | 0.0% | 1.6% |
| Coverage 2.7-2.6 | 99.5% | 18.9% | **2.0%** | 100% | 0.4% | 1.2% |
| Coverage 2.6-2.5 | 99.5% | 13.3% | **2.4%** | 100% | 0.0% | 0.0% |

Two distinct, opposite-direction effects, netting out to no change:

1. **LFP's magnitude mismatch improved substantially** -- synthetic LFP's
   mean dV/dQ in the target zone moved from 2-6x larger than real
   (experiment 16) to within ~2x and often closer (e.g. bin 3.0-2.9: real
   -0.31, synthetic -0.71, vs. experiment 16's -1.91). Consistent with
   the hypothesis: gentler discharge reduces IR-drop-driven curve
   steepness, bringing synthetic LFP's shape closer to real. **But LFP's
   coverage in the target zone got *worse*, not better** (13.3% down to
   2-13% across bins, vs. 13-56% at 0.6C) -- at a much lower C-rate, the
   discharge takes far more capacity to reach the same voltage, and the
   solver's adaptive stepping ends up leaving fewer, not more, samples in
   this particular narrow voltage window relative to the full trace.
2. **NMC's near-total absence from the target zone did not change at
   all** -- still 0-1.6% synthetic coverage vs. real's ~100%, statistically
   indistinguishable from experiment 16's 0-0.9%. The C-rate hypothesis
   does not hold for NMC: this experiment's much gentler discharge did
   not cause PyBaMM's adaptive solver to sample the 2.5-3.0V window more
   densely for NMC. Since NMC is the chemistry whose coverage gap
   dominates the accuracy collapse (LFP's own coverage, however small,
   still leaves *some* real training signal; NMC's leaves essentially
   none), fixing LFP's magnitude and hurting its coverage was a wash, and
   the real blocker -- NMC -- was untouched.

## Conclusion

**The C-rate/SOH hypothesis is refuted for the NMC sampling-density
problem specifically.** Whatever governs how densely PyBaMM's adaptive
solver samples the 2.5-3.0V window for NMC, it is not primarily a
function of discharge rate or state of health -- it's more likely tied to
the NMC parameter sets' (Chen2020/Mohtat2020/OKane2022) intrinsic
open-circuit-voltage curve shape in that specific voltage range (e.g. a
relatively flat segment that the adaptive solver can cross in few large
steps regardless of how slowly current is drawn). A fix would need to
target that directly -- e.g. forcing an explicit finer voltage/time grid
restricted to this window (untested; experiment 13 found forcing a finer
*global* time grid made the unrelated termination artifact *worse*, but
that finding was about the cutoff-adjacent behavior specifically, not
mid-curve sampling density away from it, so it doesn't settle this
question either way) -- rather than adjusting discharge conditions.

LFP's magnitude improvement at low C-rate is a genuine, real finding on
its own, independent of this experiment's main question, and could be
worth revisiting for a future test that isn't bottlenecked by NMC's
missing target-zone data.

See `../16_low_voltage_only_sim_to_real/RESULTS.md` for the original
diagnosis this experiment tested against.
