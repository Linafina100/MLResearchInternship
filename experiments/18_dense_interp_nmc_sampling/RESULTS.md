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

## Conclusion

`t_interp` is a real, mechanistically sound technique -- verified to add
genuine (not fabricated/interpolated-through-a-discontinuity) dense
samples in the target zone in the cases where the underlying dynamics
are smooth enough to have them. But since root `simulate_batteries.py`
pools all 3 NMC parameter sets and SOH is meant to span a *degraded*
range (0.50-0.85 elsewhere in this project) for the actual recycling use
case, a full-scale application of this fix would, at best, only recover
coverage for the Mohtat2020-parameterized third of NMC batteries (with a
new outlier-handling problem to solve), while Chen2020/OKane2022-
parameterized batteries (the other two-thirds) would remain exactly as
uncovered as in experiment 16. **This does not fully close experiment
16's coverage gap on its own.**

Two honest paths forward, not attempted here:
1. Apply `t_interp` only where it demonstrably helps (Mohtat2020, any
   SOH) and separately solve the outlier-handling question for it, while
   accepting Chen2020/OKane2022 stay uncovered -- a partial, quantifiable
   improvement rather than a full fix.
2. Revisit candidate C from the plan-mode investigation preceding this
   experiment (different/custom NMC parameter sets or calibrated OCP
   curves) specifically for Chen2020/OKane2022's degraded-SOH regime,
   since `t_interp` cannot help where the model's own physics produces a
   genuine near-discontinuity.

Full per-combination data: `diagnostic_results.csv`.
