# Does finer discharge sampling fix the termination-step artifact? (No.)

## What this experiment tests

Experiment 11 found a solver-termination artifact: every battery's last
raw sample lands within 0.02V of its voltage cutoff via one oversized
final step, and dividing `dV` by the resulting `dQ` produces exploded
values that contaminate two feature bins in experiment 03's low-SOC data
(quantified in experiment 12: ~20 accuracy points at SOC 0.1-0.4).
Experiment 12's fix excludes those bins outright. This experiment asks
whether the artifact can instead be avoided at the source -- by making
`simulate_batteries.py` output the discharge trace on a finer, explicit
time grid, so the final step is never abnormally large in the first
place, with zero data discarded anywhere.

PyBaMM experiment steps accept a `(<duration> period)` suffix
(`"Discharge at 1.2 A until 2.3 V (1 second period)"`) that forces output
onto a regular grid instead of the solver's own adaptive/default spacing
-- `simulate_batteries.py` currently leaves this unset. `test_period_resolution.py`
reruns the exact per-battery randomization `simulate_batteries.py` uses
(chemistry pooling, capacity scaling, SOH, resistance/temperature noise),
restricted to SOC 0.1-0.4 (the worst-affected interval), at four period
settings, and compares the final raw transition's dV/dQ across them. No
full sweep re-simulation was needed -- the batch result below is decisive
on its own.

## Results

20 variations per chemistry per period setting (seeded, same
randomization across period settings so only the period differs):

| Period | Chemistry | Median dV/dQ | Max &#124;dV/dQ&#124; | n(&#124;dV/dQ&#124;>100) |
|---|---|---|---|---|
| default (no period) | LFP | -20.4 | 304.8 | 4/20 |
| default (no period) | NMC | -140.5 | 1,869.1 | 11/18 |
| 10 seconds | LFP | -24.6 | 708.9 | 7/20 |
| 10 seconds | NMC | -378.1 | 4,112.2 | 18/18 |
| 2 seconds | LFP | -25.0 | 6,049.4 | 7/20 |
| 2 seconds | NMC | -2,039.6 | 35,721.9 | 18/18 |
| 1 second | LFP | -25.1 | 6,049.4 | 7/20 |
| 1 second | NMC | -4,012.5 | 35,721.9 | 18/18 |

**Finer sampling makes the artifact monotonically worse, not better.**
NMC's median final-transition dV/dQ goes from -140 (default) to -4,013
(1-second period) -- almost 30x worse -- and NMC hits 100% of batteries
over the |dV/dQ|>100 threshold at every explicit period setting, up from
61% by default. LFP shows the same direction of effect, less severely
(its cutoff sits further from its curve's steep region).

## Why this happens

PyBaMM's period mechanism builds an evenly-spaced output grid from t=0 to
a placeholder final time, then reports the solution at each grid point up
to and including the moment the voltage-cutoff event actually fires. The
*last* reported interval is therefore whatever *partial* period remains
between the last full grid point and the event -- not a full period, and
not bounded below. A finer period doesn't just add more points leading up
to the cutoff; it also shrinks this final leftover interval, which can be
arbitrarily small. Since the underlying voltage-vs-capacity slope right at
the cutoff appears to be genuinely steep (concentration polarization
approaching a bound as the cell nears full depletion -- plausibly *why*
cutoffs are set where they are, to stay clear of this region), a smaller
`dQ` denominator over that same steep true slope produces a *larger*
dV/dQ estimate, not a smaller one. Finer sampling narrows the window used
to estimate a derivative that appears to diverge as the window shrinks --
the opposite of what would fix a genuine under-resolution problem.

This also explains why the two failure modes in this repo look similar
but aren't: experiment 04's pulse-protocol sampling-resolution problem
(fixed unsuccessfully by experiment 05's interpolation, but a genuine
under-resolution issue) was about *missing* a region entirely between two
widely-spaced pulses. This artifact is the opposite -- a region that's
*always* sampled, at the one place (the cutoff boundary) where doing so
with a plain finite difference is inherently unstable, regardless of how
finely you sample.

## Conclusion

**Simulation-level resampling does not fix this artifact, and reliably
makes it worse.** This rules out the "just sample the discharge curve more
finely" approach as a category. `simulate_batteries.py` is unmodified by
this experiment -- no production change was warranted given the negative
result. The correct category of fix remains excluding the affected
bins/transitions after the fact, as experiment 12 already does (or, not
attempted here, excluding samples from each battery's final N raw points
*before* binning at the feature-engineering level, which was the
originally proposed alternative to experiment 12's bin-level exclusion and
remains untested).
