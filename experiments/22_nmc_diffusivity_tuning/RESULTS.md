# Experiment 22: tune PyBaMM NMC parameters to fix the dV/dQ magnitude mismatch

## Motivation

Experiments 20-21 closed the NMC target-zone coverage gap but left a
3-6x systematic dV/dQ magnitude mismatch (synthetic steeper than real)
as the sole remaining blocker; experiment 21 confirmed post-hoc feature
normalization can't fix it. This experiment investigates fixing it at
the source, in PyBaMM's own parameters, rather than post-hoc.

## Plan-mode investigation: which physical parameter actually governs curve steepness

Before building anything, three candidate mechanisms were tested
directly (single controlled battery: Chen2020, SOH=0.8 with experiment
20's SOH-scaling fix, 0.15C, resistance_factor=0.8, dense `t_interp`
output):

| Candidate | Mechanism | Result |
|---|---|---|
| Positive electrode conductivity / Contact resistance (internal resistance) | Ohmic drop | **No effect at all** -- identical output to the decimal at every value tested |
| Positive electrode exchange-current density (reaction rate) | Activation overpotential | **Weak** -- barely moves the needle (-7.97 to -7.45 across 0.5x-0.1x) |
| **Positive particle diffusivity** | Li+ diffusion inside NMC particles -- slower diffusion smears the voltage transition over more capacity | **Works**: -8.02 -> -5.92 at 10x reduction |

**Why resistance doesn't work, and diffusivity does**: a resistive term
subtracts `I*R` from terminal voltage. Under ~constant-current
discharge, `I*R` is nearly constant, so it shifts the *entire* curve by
a fixed offset -- it cannot change the curve's *slope*, hence cannot
change dV/dQ (plain SPM doesn't even wire electrode conductivity into
the solved dynamics at all). Diffusivity changes how fast the particle
*surface* concentration (which sets the OCP) tracks the *bulk*
concentration (which sets remaining capacity) -- this directly reshapes
the curve's local steepness, which is the actual mechanism needed.

A quick single-point sweep also found the safe window is narrow: 10x
reduction worked well; 15x already degraded to a single extreme-outlier
point; 20x+ collapsed to zero target-zone points entirely (the same
kind of collapse experiments 18/20 characterized, now diffusivity-driven
instead of SOH-driven).

## Phase 1: systematic sweep across the real operating range

`diagnose_diffusivity_factor.py` swept diffusivity reduction factors
{1 (baseline), 4, 6, 8, 10, 12} x all 3 NMC parameter sets x the real
C-rate (0.1/0.15/0.2) and temperature (15/25/35C) ranges, at SOH=0.8,
using the same dense-`t_interp`-and-splice method as experiments 18/20.
(One implementation snag along the way: Chen2020 defines diffusivity as
a plain constant, but Mohtat2020/OKane2022 define it as a callable
`(sto, T) -> value` -- fixed by wrapping the callable rather than
dividing it directly.)

**Result: a clean, robust factor for two of the three parameter sets --
and a clear negative result for the third.**

| Parameter set | factor=1 (baseline) | factor=10 |
|---|---|---|
| Chen2020 | mean -8.04, 3/9 in real target, 0 outliers | **mean -5.81, 9/9 in real target (-8 to -3), 0 outliers** |
| OKane2022 | mean -9.71, 0/9 in real target, 0 outliers | **mean -6.27, 9/9 in real target, 0 outliers** |
| Mohtat2020 | mean -27.76, 0/9 in real target, 9 outliers | mean -26.45, 0/9 in real target, 9 outliers (unchanged) |

Chen2020 and OKane2022 both converge on **factor=10** as the largest
reduction that stays robust across the full swept space (factor=12
already drops to 6/9 robust combinations for both -- some C-rate/
temperature combinations collapse to zero target-zone points).

**Mohtat2020 is not fixed by this lever at all** -- its mean magnitude
barely moves (-27.76 -> -26.45) across the entire factor range 1-12,
with the same 9 outlier combinations throughout. Consistent with
Mohtat2020's established pattern (experiments 07/18) of behaving
fundamentally differently from the other two NMC parameter sets;
diffusivity is evidently not its dominant lever, and this experiment
doesn't identify what is.

**User decision**: rather than diluting this experiment's result with
an unfixed 1/3 of the NMC pool, or pausing to investigate Mohtat2020
separately, Mohtat2020 is dropped from the NMC parameter-set pool for
experiment 22's Phase 2. It remains a known, separate open problem for
future work, not resolved here.

Full per-combination data: `diagnostic_results.csv`.

## Phase 2: full pipeline with Chen2020/OKane2022 + diffusivity/10

`simulate_batteries_diffusivity_tuned.py`: SOH=0.8 (fixed), C-rate
0.1-0.2, 1.5V cutoff, dense `t_interp` output, NMC pool restricted to
Chen2020+OKane2022 (Mohtat2020 dropped), positive particle diffusivity
/10. 498/498 simulations succeeded (matching experiment 20's clean run).

### Result: the first genuine (non-chance) sim-to-real signal in this entire investigation -- for Random Forest, not XGBoost

| Model | Raw accuracy | **Balanced accuracy** | LFP recall | NMC recall |
|---|---|---|---|---|
| **Random Forest** | 74.65% | **82.92%** | 0.991 | 0.667 |
| XGBoost | 28.78% | 52.60% | 0.993 | 0.059 |

Unlike experiment 21's illusory jumps, this is real: 82.92% balanced
accuracy is far above chance (50%), and LFP/NMC recall are both
reasonably high (not one collapsed to near-zero while the other sits
near 1.0, the signature of the class-imbalance artifact experiment 21
exposed). **This is the first non-chance balanced-accuracy result
anywhere in experiments 16-22.**

XGBoost, however, still fails -- balanced accuracy 52.60% (barely above
chance), NMC recall 0.059 (collapsed to predicting almost-always-LFP,
the same degenerate pattern seen throughout this investigation). This
RF-vs-XGBoost asymmetry under train/test distribution shift matches a
pattern already documented in experiment 07: Random Forest's bagged,
averaged-across-many-trees structure is more robust to this kind of
shift than XGBoost's sequential boosted splits, which tend to fit
decision thresholds too specifically to the synthetic training
distribution's exact value range.

### Why it's not a complete fix: the magnitude match is uneven across the target zone

| Bin | Real NMC | Synth NMC (exp20, no fix) | Synth NMC (exp22, diffusivity fix) | Real LFP | Synth LFP (exp22) |
|---|---|---|---|---|---|
| 3.0-2.9 | -3.36 | -9.90 | **-3.57** (near-exact match) | -0.31 | -1.88 |
| 2.9-2.8 | -2.73 | -13.63 | -5.95 (better, still ~2x) | -0.72 | -2.83 |
| 2.8-2.7 | -3.30 | -17.62 | -10.74 (still ~3x) | -1.18 | -4.65 |
| 2.7-2.6 | -5.24 | -26.22 | -14.51 (still ~3x) | -1.34 | -6.66 |
| 2.6-2.5 | -7.73 | -16.17 | -16.74 (still ~2x, unimproved) | -2.87 | -8.72 |

The fix works best at the *top* of the target zone (3.0-2.9V: synthetic
NMC now matches real almost exactly) and progressively less well
deeper into the zone -- the deepest bins (2.7-2.6, 2.6-2.5) are barely
better than experiment 20's unfixed baseline. NMC coverage also dips
slightly in the deepest bins (0.892, 0.831 vs. ~1.0 elsewhere) --
diffusivity/10 trades a small amount of coverage at the very bottom for
a much better magnitude match higher up. **LFP's own pre-existing
magnitude mismatch (documented since experiments 07/16/17) is untouched
here** -- this experiment only tuned NMC's diffusivity, per its scope.

## Conclusion

Tuning NMC's solid-state diffusivity at the source, rather than any
post-hoc feature transform, produces the first genuine sim-to-real
signal found across this entire coverage-then-magnitude investigation
(experiments 16-22) -- but only for Random Forest, and only partially:
the fix works very well at the top of the target zone and progressively
less well deeper into it, and LFP's own magnitude mismatch remains
completely unaddressed. Two natural next steps, neither attempted here:
(1) investigate why XGBoost doesn't benefit the way Random Forest does
despite seeing the same improved features -- possibly worth deliberately
using Random Forest as the primary model for this feature set rather
than treating XGBoost's failure as a blocker; (2) a finer-grained or
depth-dependent diffusivity adjustment (rather than one flat factor for
the whole positive electrode) to close the remaining gap in the deepest
bins, and/or the analogous investigation for LFP's own diffusivity
parameter (Prada2013) to address its separate, still-unfixed mismatch.

Full run outputs: `sim_and_real_raw.csv`, `features/ml_features.csv`
(gitignored, regenerable via `simulate_batteries_diffusivity_tuned.py`
then `evaluate_diffusivity_tuned_sim_to_real.py`).
