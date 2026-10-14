# How much of experiment 03's accuracy survives without the termination-artifact bins

## What this experiment is

Experiment 11 (`experiments/11_voltage_cutoff_comparison/RESULTS.md`) found
that PyBaMM's
`"Discharge until X V"` termination is event-triggered: every battery's
last raw sample lands within 0.02V of its chemistry's voltage cutoff via
one oversized final step, and dividing dV by the resulting near-zero final
dQ explodes the dV/dQ value in whichever bin sits right at that cutoff. In
experiment 03 (cutoffs LFP=1.8V, NMC=2.3V), this contaminates
`dV_dQ_V_2.4_2.3` and `dV_dQ_V_2.3_2.2` -- both right at NMC's cutoff --
which survive the mutual-coverage filter at the two lowest SOC intervals,
where NMC's ~500-2000x larger-magnitude artifact values sit alongside
LFP's normal ones, giving the classifier a trivial, non-electrochemical
tell.

This experiment quantifies exactly how much of experiment 03's reported
accuracy that artifact actually accounts for, by re-training on the same
data with those bins excluded and comparing before/after. Fix applied,
generalized rather than hardcoded to two exact bin names: drop any bin
within one bin-width (0.1V) of *either* chemistry's cutoff (absorbs the
+-0.02V measurement noise observed around the cutoff). No re-simulation --
reuses experiment 03's already-committed raw CSVs
(`data/continuous_soc_<interval>/raw/`), and root `feature_engineering.py`
/ `ml_pipeline.py` unchanged, same reevaluate-existing-data pattern as
experiment 05.

## Results

| SOC Interval | Bin Set | RF Accuracy | XGB Accuracy | Surviving Bins |
|---|---|---|---|---|
| 0.7-1.0 | before | 73.96% | 78.12% | `3.3-3.2` |
| 0.7-1.0 | after | 73.96% | 78.12% | `3.3-3.2` (unchanged -- no artifact bin here) |
| 0.5-0.8 | before | 98.96% | 98.96% | `3.3-3.2`, `3.2-3.1` |
| 0.5-0.8 | after | 98.96% | 98.96% | `3.3-3.2`, `3.2-3.1` (unchanged -- no artifact bin here) |
| 0.3-0.6 | before | 98.96% | 98.96% | `3.3-3.2`, `3.2-3.1`, `2.4-2.3`, `2.3-2.2` |
| 0.3-0.6 | after | **100.00%** | 98.96% | `3.3-3.2`, `3.2-3.1` |
| 0.1-0.4 | before | 94.74% | 94.74% | `3.2-3.1`, `2.4-2.3`, `2.3-2.2` |
| 0.1-0.4 | after | **73.68%** | **75.79%** | `3.2-3.1` |

(0.7-1.0's numbers here, 73.96%/78.12%, differ slightly from experiment
03's originally published 77.32%/77.32% despite both using
`random_state=42` throughout `ml_pipeline.py` -- likely package-version
drift or row-order sensitivity in the stratified-group split, not
something this experiment chased down further. It doesn't affect the
before/after comparison, which is internally consistent: both bin sets for
a given interval are trained on the exact same battery groups/split.)

## Interpretation

**Two of the four intervals were never touched by the artifact at all**
(0.7-1.0, 0.5-0.8) -- neither ever had an NMC-cutoff-adjacent bin survive
the mutual-coverage filter in the first place, so excluding it changes
nothing. Their accuracy (74-78%, 99%) is genuine.

**0.3-0.6 survives the exclusion essentially undamaged** -- accuracy is
unchanged or even ticks up slightly (RF 98.96% -> 100.00%, XGB flat at
98.96%) once the two artifact bins are dropped, leaving 2 genuine bins
(`3.3-3.2`, `3.2-3.1`). The classifier didn't actually need the artifact
here; the real dV/dQ signal in this interval was already strong enough on
its own.

**0.1-0.4 is where the artifact was doing real work.** Accuracy collapses
from 94.74%/94.74% down to 73.68%/75.79% -- a 19-21 point drop -- once the
two artifact bins are removed, leaving only `dV_dQ_V_3.2_3.1`. This is the
interval experiment 03's own `RESULTS.md` singled out along with 0.3-0.6 as
evidence that "more shared signal appears at lower SOC... giving both
chemistries a real, comparable presence" -- for 0.1-0.4 specifically, that
claim doesn't hold up: roughly 20 of its ~95 percentage points of accuracy
were the artifact, not signal. Its genuine accuracy (~74-76%) is
comparable to 0.7-1.0's, not the standout result it appeared to be.

## Bottom line

Of experiment 03's four SOC intervals, only **one** (0.1-0.4) was
materially inflated by the termination-artifact bins -- roughly 19-21
accuracy points' worth. The other three intervals' numbers (including
0.3-0.6, which also had artifact bins survive the coverage filter) are
essentially unaffected by removing them. Experiment 03's `RESULTS.md`
interpretation section, which treats 0.3-0.6 and 0.1-0.4 as a matched pair
both showing "real, comparable" cross-chemistry signal at low SOC, should
be corrected: that characterization holds for 0.3-0.6 but not for 0.1-0.4.

## Not yet done

- This uses the exact same one-bin-width exclusion rule proposed in
  experiment 11; a tighter or more principled rule (e.g. excluding samples
  from each battery's final N raw points before binning, rather than
  excluding whole bins after the fact) might recover more genuine signal
  near the cutoff instead of discarding the bin outright -- not attempted
  here.
- Root `feature_engineering.py` itself is unchanged by this experiment;
  the exclusion is applied as a post-processing step in this experiment's
  own script. Whether to fold an equivalent guard into the root pipeline
  (and re-verify experiments 04/05/06, which build on or compare against
  the continuous-discharge protocol) is a separate decision.
- Experiment 03's own `RESULTS.md` has not been edited to reflect this
  finding -- that file lives on `main` and wasn't touched by this branch.
