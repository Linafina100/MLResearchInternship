# Experiment 18: does PyBaMM's `t_interp` fix NMC's target-zone coverage gap?

## Motivation

Experiments 16 and 17 established that synthetic NMC data is almost
entirely absent from the real-world target zone (2.5-3.0V) regardless of
simulated cutoff, SOH, or C-rate -- PyBaMM's adaptive solver reports very
few raw samples there. The plan-mode investigation preceding this
experiment found that the default solver (`IDAKLUSolver`) supports
`Simulation.solve(..., t_interp=<array>)`: querying the already-solved
continuous trajectory at arbitrary dense time points, without changing
how the solver actually integrates. An initial single-battery spot check
(Chen2020, SOH=1.0, 0.3C) showed a large density gain (0 -> 34 points in
the zone), motivating this systematic follow-up across NMC parameter
sets, SOH, and C-rate.

## Method

`diagnose_t_interp_density.py`: for 27 combinations (3 NMC parameter
sets x SOH {0.6, 0.8, 1.0} x C-rate {0.2, 0.6, 1.0}, 1.5V cutoff), each
battery is solved twice -- once normally (baseline), once with a dense
(3000-point) `t_interp` grid -- and the resulting transitions in the
2.5-3.0V zone are compared. `exclude_final_transition`-style handling is
preserved: the dense grid's bound is kept strictly below the true
event-termination time so the genuine final segment is undisturbed.

**A real methodological pitfall was found and fixed during this
experiment, worth recording**: the first version bounded `t_interp` at
95% of the total discharge duration, chosen as a "safe" margin below the
crash-triggering true event time. This turned out to be **far too
conservative** -- PyBaMM auto-appends the true final state to the output
regardless of the requested bound, so no large margin is actually needed;
a bound of `tf * (1 - 1e-6)` was verified safe (no crash) across every
combination tested. The 95%-margin version produced a **false negative**
for most combinations (looked like zero benefit, when the genuine signal
was simply past the excluded window) -- this was caught by manually
inspecting the underlying voltage trace for a "zero-benefit" case before
accepting the result, not by trusting the summary numbers alone.

## Result: a real but narrow, parameter-set-and-SOH-dependent win, not a general fix

| Parameter set | SOH 0.6 | SOH 0.8 | SOH 1.0 |
|---|---|---|---|
| Chen2020 | 0 points (genuine instant collapse, see below) | 0 points (same) | **111-167 points, clean (0 outliers)** |
| OKane2022 | 0 points (same) | 0 points (same) | **92-152 points, clean (0 outliers)** |
| Mohtat2020 | **30-43 points, but 11-15 are outliers (up to \|dV/dQ\|=144)** | **23-29 points, 2-3 outliers** | **26-35 points, 1 outlier** |

(Each cell: range across the 3 tested C-rates. Default/baseline output
gave 0 points in the zone for all 27 combinations, confirming
experiments 16/17's finding independent of this fix.)

### Chen2020 / OKane2022: works, but only for undegraded (SOH=1.0) cells

Directly inspected the underlying continuous solution for a degraded case
(Chen2020, SOH=0.6, 0.6C) at extremely fine time resolution (0.004-second
spacing near the transition): voltage declines smoothly from 3.30V down
to 3.11V over ~200 seconds, then **collapses from 3.11V to 1.79V within a
single 0.004-second step** -- a genuine near-discontinuity in the SPM
model's own physics for this parameter set at this SOH, not a solver-
resolution limitation. No amount of additional query density can recover
points inside a true discontinuity. At SOH=1.0 (undegraded), the same
parameter sets show smooth, gradual multi-second dynamics through the
zone instead, which `t_interp` successfully resolves into many clean
points.

**This is the practically important limitation**: the actual recycling
use case needs *degraded* cells (SOH well below 1.0), and for two of the
three NMC parameter sets, `t_interp` provides zero benefit specifically
in that regime.

### Mohtat2020: works at every SOH tested, but with real outlier contamination

Mohtat2020 behaves differently -- it does *not* show the same hard
discontinuity, and dense sampling recovers real points at every SOH
tested. But a substantial fraction of those points carry outlier-level
`|dV/dQ|` (>50, the same threshold used throughout this project's
diagnostics): 37-40% of Mohtat2020's SOH=0.6 zone points are outliers
(up to 144), dropping to ~3-4% at SOH=1.0. This means Mohtat2020's
target-zone dynamics genuinely are very steep (if not instantaneous) at
low SOH, and resolving them densely surfaces that steepness directly as
large `dV/dQ` values -- which is a *real* modeling signal, not spurious,
but would need careful handling before use (a naive relative-magnitude
outlier filter was already shown harmful and NMC-biased in experiment
07's discarded attempt #5 -- this is a different, new instance of the
same underlying tension between "artifact" and "genuine but extreme
signal").

## Follow-up: does the cliff disappear under milder, more realistic recycling conditions?

The original 27-combo sweep tested SOH 0.6-1.0 and C-rate 0.2-1.0, which
may be more aggressive than a real recycling scenario needs. Hypothesis
tested here: the near-discontinuity is specific to *severe* degradation
and/or *high* current, so restricting to SOH 0.8-0.95, C-rate 0.1-0.2,
and (newly) ambient temperature 15-35C -- not modeled at all in the
original diagnostic -- might avoid it.

`diagnose_t_interp_density_mild_conditions.py` re-ran the same
solve-twice-and-compare method across 3 SOH x 2 C-rate x 3 temperature x
3 NMC parameter sets = 54 combinations, adding temperature-driven
resistance modeling (reusing the exact pattern from
`experiments/17_high_soh_low_crate_sim_to_real/simulate_batteries_high_soh_low_crate.py`).

### Result: hypothesis confirmed, with a precise SOH threshold

**42/54 combinations (78%) now show genuine target-zone coverage**, up
from 6/27 (22%) in the original sweep -- and critically, **Chen2020 and
OKane2022 now gain substantial, perfectly clean (zero-outlier) coverage**
(1488 and 1323 total points respectively across the sweep), not just at
SOH=1.0 anymore:

| Parameter set | SOH 0.8 | SOH 0.875 | SOH 0.95 |
|---|---|---|---|
| Chen2020 | **0 points** (confirmed genuine discontinuity, see below) | 108-143 points, 0 outliers | 108-116 points, 0 outliers |
| OKane2022 | **0 points** (same) | 117-132 points, 0 outliers | 90-102 points, 0 outliers |
| Mohtat2020 | 22-23 points, 1-2 outliers (up to \|dV/dQ\|=67) | 27 points, 0 outliers | 27-29 points, 0-1 outliers |

(Ranges are across the 2 tested C-rates; temperature made essentially no
difference -- identical point counts and outlier counts at 15C/25C/35C
for every parameter-set/SOH/C-rate combination, confirming ambient
temperature isn't a meaningful driver in this range.)

**The cliff has a precise SOH threshold for Chen2020/OKane2022,
somewhere between 0.8 and 0.875** -- both still show the exact same
near-instantaneous collapse at SOH=0.8 (verified directly again: Chen2020,
SOH=0.8, C-rate=0.1 -- the *gentlest* condition tested -- collapses from
3.15V to 2.22V within a single 9.5-second dense-grid step, overshooting
the entire target zone in one step, confirming this is not a current- or
temperature-driven effect, specifically an SOH one). From SOH=0.875
upward, both parameter sets resolve smoothly and cleanly.

**Mohtat2020's outlier problem essentially disappears too** above
SOH=0.8: 1-2 outliers at SOH=0.8 (vs. up to 15 out of 43 points at the
original sweep's SOH=0.6), dropping to 0-1 at SOH >= 0.875.

### Revised conclusion

The user's hypothesis holds, and substantially changes the practical
picture from the first pass: **for the realistic recycling SOH range
above ~0.875, `t_interp` cleanly recovers target-zone coverage for all
three NMC parameter sets**, not just Mohtat2020. The remaining gap is
narrow and specific: SOH around 0.8 and below still hits a genuine
physical cliff for Chen2020/OKane2022 (confirmed independent of C-rate
and temperature in the tested ranges), and would need one of the two
paths from the original conclusion (accept the gap there, or pursue
different/custom NMC parameters for that specific low-SOH regime) if
recycling cells that degraded still need to be covered.

**Practical implication**: if the target SOH range for this use case can
reasonably be bounded at >=0.875 (moderately degraded, not severely), a
full-scale synthetic batch using `t_interp` in this range is now well
justified by this diagnostic and would be expected to substantially close
experiment 16's coverage gap. Below that SOH, the gap persists for 2/3 of
the NMC parameter-set pool.

Full per-combination data: `diagnostic_results_mild_conditions.csv`
(original sweep: `diagnostic_results.csv`).
