# Experiment 20: closing the sim-to-real coverage gap, a normalization dead end, and a working NMC diffusivity fix

This experiment folder consolidates three chronological phases of the
same investigation (originally run as separate branches/experiment
numbers during development; folded into one numbered experiment here so
the project's experiment sequence on `main` stays contiguous). Each
phase's own scripts are kept in full below; only the experiment numbers
in the surrounding narrative have been consolidated.

## Part A: SOH-scaling fix + t_interp closes the coverage gap at SOH=0.8

### Motivation

Experiment 18 (+ its mild-conditions follow-up) found `t_interp` cleanly
recovers synthetic NMC's missing 2.5-3.0V target-zone samples for
SOH>=0.875, but hits a genuine near-instantaneous voltage collapse for
Chen2020/OKane2022 at SOH=0.8 -- the real EV-retirement threshold this
use case needs covered. Plan-mode investigation traced the exact cause:
every simulate script in this project scales `Maximum concentration in
negative/positive electrode` by SOH (reduced storage capacity) without
correspondingly scaling `Initial concentration` (the lithium inventory
that capacity has to hold), forcing the positive electrode to overfill
past its own stoichiometry limit once SOH drops to ~0.875 or below.
Scaling both together keeps stoichiometry (a ratio) invariant --
verified directly (electrode balance identical across every SOH once
both are scaled) and in an actual discharge solve (zero single-step
jumps >0.3V at SOH=0.8 for Chen2020/OKane2022, vs. a ~1.3V collapse
before).

### A mistake caught and fixed mid-experiment, worth recording

The first version of the SOH-scaling-fix simulate script applied the
SOH-scaling fix but used PyBaMM's **default** adaptive solver output
(`sim.solve(initial_soc=soc)`), the same as every other simulate script
in this project. Result: **no improvement at all** -- 24.33% accuracy,
identical to experiments 16/17, NMC recall still 0.00. Checking the raw
output confirmed why: only 37-75 rows per battery, the same sparse
default density experiments 16-18 already diagnosed as the *separate*
coverage problem. **The SOH-scaling fix smooths the underlying physics
(no more discontinuity) but does not, by itself, make the default
adaptive solver report more points in the target zone** -- that
requires `t_interp` on top, which is the entire subject of experiment
18. Applying only one of the two necessary fixes reproduced the
original null result exactly.

Fixed by adding experiment 18's validated `t_interp` dense-solve-and-
splice method (`solve_dense()`) to every battery's solve, on top of the
SOH-scaling fix -- not instead of it. Both fixes are necessary; neither
alone is sufficient.

### Method (corrected version)

SOH fixed at exactly 0.8 (not randomized, for a clean read at the
specific threshold), C-rate randomized 0.1-0.2 per battery, ambient
temperature randomized 15-35C, 1.5V cutoff, all 3 NMC parameter sets
pooled -- otherwise identical to prior scripts, with both fixes applied
per battery. 498/498 simulations succeeded (the cleanest run yet at the
time; even better than experiment 17's 497/498, since fixing the
stoichiometry overflow also incidentally removes the unrelated `Q_Li`-
out-of-range failure mode seen in earlier experiments). Raw output: 3001
rows/battery (dense), vs. 37-75 with the default solver.

### Result: the coverage gap is closed; a different, already-known problem is now the limiter

| Model | Accuracy |
|---|---|
| Random Forest | **27.81%** |
| XGBoost | 25.82% |

Modest in absolute terms, but real: NMC recall moved off zero for the
first time in this entire investigation (0.00 -> 0.04). Precision on
NMC is 0.98 -- when the model does predict NMC, it's almost always
right; it just rarely predicts NMC at all.

**Coverage is now essentially solved**:

| Bin | Real LFP cov. | Synth LFP cov. | Real NMC cov. | Synth NMC cov. |
|---|---|---|---|---|
| 3.0-2.9 | 99.5% | **99.6%** | 99.4% | **100%** |
| 2.9-2.8 | 99.5% | **99.6%** | 100% | **100%** |
| 2.8-2.7 | 99.5% | **99.6%** | 100% | **100%** |
| 2.7-2.6 | 99.5% | **99.6%** | 100% | **100%** |
| 2.6-2.5 | 99.5% | **99.6%** | 100% | 73.5% |

Compare to experiment 16/17's 0-1.6% synthetic NMC coverage -- this is a
complete turnaround on the specific problem those experiments
identified.

**But a different, already-known problem now dominates**: with real
coverage to finally measure against, a systematic magnitude mismatch is
clearly visible -- the same pattern experiment 07 first documented for
the full curve:

| Bin | Real LFP | Synth LFP | Real NMC | Synth NMC |
|---|---|---|---|---|
| 3.0-2.9 | -0.31 | -1.88 | -3.36 | -9.90 |
| 2.9-2.8 | -0.72 | -2.84 | -2.73 | -13.63 |
| 2.8-2.7 | -1.18 | -4.64 | -3.30 | -17.62 |
| 2.7-2.6 | -1.34 | -6.67 | -5.24 | -26.22 |
| 2.6-2.5 | -2.87 | -8.72 | -7.73 | -16.17 |

Synthetic dV/dQ runs 3-6x larger than real for both chemistries in this
zone -- the same systematic gap this project has documented before
(experiment 07's original diagnosis, experiment 16/17's LFP findings).

**Conclusion**: the SOH-scaling + `t_interp` combination fully solves
the sampling-density/coverage problem this investigation had been
chasing, specifically at the realistic SOH=0.8 threshold this use case
needs. What remains is the systematic dV/dQ magnitude mismatch -- the
subject of Parts B and C below.

**Note on the broader implication**: this SOH-scaling fix (scale
`Initial concentration` alongside `Maximum concentration`) was
deliberately *not* retrofitted into any prior experiment's scripts, per
explicit instruction -- earlier SOH-driven results elsewhere in this
project were generated without this fix and should be understood in
that light. This is recorded as a methodological finding for the
project write-up rather than applied retroactively.

## Part B: per-battery normalization -- a dead end, and a cautionary tale

### Motivation

Part A closed the NMC target-zone coverage gap but exposed the 3-6x
systematic dV/dQ magnitude mismatch. Since absolute voltage bin
*identity* can't be normalized away (which bins a chemistry reaches is
itself real signal, established since experiment 07), this phase tested
a different axis: per-battery (row-wise) normalization of the dV/dQ
feature vector across the 5 target-zone bins, on the reasoning that if
real and synthetic cells differ mainly in scale but share a similar
relative shape, removing each battery's own scale before training might
let the classifier learn from shape alone. Reused Part A's already-
simulated data; only the ML step changed.

### Method

Tested four variants: `none` (Part A's baseline), `l2` (unit-norm each
battery's feature vector), `minmax` (rescale each battery's own 5
values to [0,1]), `standardize` (z-score each battery's own 5 values).
All NaN-aware. Otherwise identical pipeline to Part A (root
`ml_pipeline.py`, same percentile-clip/impute/StandardScaler
preprocessing, same RF/XGBoost hyperparameters).

### Result: raw accuracy looked dramatic; it is not real

| Variant | Model | Raw accuracy | **Balanced accuracy** | LFP recall | NMC recall |
|---|---|---|---|---|---|
| none (Part A baseline) | RF | 27.81% | 52.07% | 0.997 | 0.045 |
| none | XGB | 25.82% | 50.87% | 1.000 | 0.017 |
| l2 | RF | **73.17%** | 50.15% | 0.050 | 0.953 |
| l2 | XGB | **72.74%** | 50.04% | 0.055 | 0.946 |
| minmax | RF | 62.06% | 52.14% | 0.327 | 0.716 |
| minmax | XGB | **75.46%** | 50.32% | 0.010 | 0.996 |
| standardize | RF | 52.69% | 35.42% | 0.016 | 0.693 |
| standardize | XGB | 2.63% | 2.15% | 0.012 | 0.031 |

The real test set is 578 LFP + 1781 NMC (2359 total) -- **75.50% NMC**.
A model that always predicts NMC scores 75.50% raw accuracy while
discriminating nothing.

**Every single variant's balanced accuracy sits at or below chance
(50%)**, several dramatically below (standardize+XGBoost: 2.15%, a
near-total inversion). The headline raw-accuracy numbers that looked
like a breakthrough (l2: 73%, minmax+XGBoost: 75%) are a direct artifact
of this class imbalance: `none` collapses to predicting almost-always-
LFP (close to experiment 07's original 24.5%-LFP floor); `l2` and
`minmax`+XGBoost instead collapse to predicting almost-always-**NMC**,
and because the real test set happens to be 75.5% NMC, that flip alone
buys a large raw-accuracy number without any genuine two-way
discrimination.

**Conclusion: per-battery normalization does not unlock sim-to-real
transfer.** It does not teach the model to read shape -- it changes
which trivial single-class default the model falls back to. Balanced
accuracy (or per-class recall) must be checked, not raw accuracy alone,
whenever the test set is imbalanced -- this phase is a direct
illustration of why (see experiment 07's discarded attempt #4 for an
earlier, similarly deceptive headline-number case). Likely explanation:
the ratio of synthetic-to-real magnitude is not constant across bins
(e.g. for NMC: bin 2.7-2.6 ratio ~5.0, bin 2.6-2.5 ratio ~2.1) -- if the
mismatch isn't a pure scalar multiple, removing scale per-battery
doesn't recover a shared shape; it can scramble whatever partial
shape-similarity existed instead. The magnitude mismatch remained
unsolved going into Part C.

## Part C: tune PyBaMM's NMC parameters to fix the magnitude mismatch at the source

### Motivation

Since post-hoc feature transforms (Part B) can't fix the magnitude
mismatch, this phase fixes it at the simulation source: tuning NMC's
physical parameters so the simulated end-of-discharge curve is less
steep, closer to real depleted NMC cells' more gradual decline.

### Plan-mode investigation: which physical parameter actually governs curve steepness

Before building anything, three candidate mechanisms were tested
directly (single controlled battery: Chen2020, SOH=0.8 with Part A's
SOH-scaling fix, 0.15C, resistance_factor=0.8, dense `t_interp` output):

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
kind of collapse experiment 18/Part A characterized, now diffusivity-
driven instead of SOH-driven).

### Phase 1: systematic sweep across the real operating range

`diagnose_diffusivity_factor.py` swept diffusivity reduction factors
{1 (baseline), 4, 6, 8, 10, 12} x all 3 NMC parameter sets x the real
C-rate (0.1/0.15/0.2) and temperature (15/25/35C) ranges, at SOH=0.8,
using the same dense-`t_interp`-and-splice method as Part A. (One
implementation snag along the way: Chen2020 defines diffusivity as a
plain constant, but Mohtat2020/OKane2022 define it as a callable
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

**User decision**: rather than diluting this phase's result with an
unfixed 1/3 of the NMC pool, or pausing to investigate Mohtat2020
separately, Mohtat2020 is dropped from the NMC parameter-set pool for
Phase 2 below. It remains a known, separate open problem for future
work, not resolved here.

Full per-combination data: `diagnostic_results.csv`.

### Phase 2: full pipeline with Chen2020/OKane2022 + diffusivity/10

`simulate_batteries_diffusivity_tuned.py`: SOH=0.8 (fixed), C-rate
0.1-0.2, 1.5V cutoff, dense `t_interp` output, NMC pool restricted to
Chen2020+OKane2022 (Mohtat2020 dropped), positive particle diffusivity
/10. 498/498 simulations succeeded (matching Part A's clean run).

#### Result: the first genuine (non-chance) sim-to-real signal in this entire investigation -- for Random Forest, not XGBoost

| Model | Raw accuracy | **Balanced accuracy** | LFP recall | NMC recall |
|---|---|---|---|---|
| **Random Forest** | 74.65% | **82.92%** | 0.991 | 0.667 |
| XGBoost | 28.78% | 52.60% | 0.993 | 0.059 |

Unlike Part B's illusory jumps, this is real: 82.92% balanced accuracy
is far above chance (50%), and LFP/NMC recall are both reasonably high
(not one collapsed to near-zero while the other sits near 1.0, the
signature of the class-imbalance artifact Part B exposed). **This is
the first non-chance balanced-accuracy result anywhere in this entire
coverage-then-magnitude investigation.**

XGBoost, however, still fails -- balanced accuracy 52.60% (barely above
chance), NMC recall 0.059 (collapsed to predicting almost-always-LFP,
the same degenerate pattern seen throughout this investigation). This
RF-vs-XGBoost asymmetry under train/test distribution shift matches a
pattern already documented in experiment 07: Random Forest's bagged,
averaged-across-many-trees structure is more robust to this kind of
shift than XGBoost's sequential boosted splits, which tend to fit
decision thresholds too specifically to the synthetic training
distribution's exact value range.

#### Why it's not a complete fix: the magnitude match is uneven across the target zone

| Bin | Real NMC | Synth NMC (Part A, no fix) | Synth NMC (Part C, diffusivity fix) | Real LFP | Synth LFP (Part C) |
|---|---|---|---|---|---|
| 3.0-2.9 | -3.36 | -9.90 | **-3.57** (near-exact match) | -0.31 | -1.88 |
| 2.9-2.8 | -2.73 | -13.63 | -5.95 (better, still ~2x) | -0.72 | -2.83 |
| 2.8-2.7 | -3.30 | -17.62 | -10.74 (still ~3x) | -1.18 | -4.65 |
| 2.7-2.6 | -5.24 | -26.22 | -14.51 (still ~3x) | -1.34 | -6.66 |
| 2.6-2.5 | -7.73 | -16.17 | -16.74 (still ~2x, unimproved) | -2.87 | -8.72 |

The fix works best at the *top* of the target zone (3.0-2.9V: synthetic
NMC now matches real almost exactly) and progressively less well
deeper into the zone -- the deepest bins (2.7-2.6, 2.6-2.5) are barely
better than Part A's unfixed baseline. NMC coverage also dips slightly
in the deepest bins (0.892, 0.831 vs. ~1.0 elsewhere) -- diffusivity/10
trades a small amount of coverage at the very bottom for a much better
magnitude match higher up. **LFP's own pre-existing magnitude mismatch
(documented since experiments 07/16/17) is untouched here** -- this
phase only tuned NMC's diffusivity, per its scope.

## Overall conclusion

Across all three parts: the coverage gap (Part A) is fully solved at
the realistic SOH=0.8 threshold; post-hoc feature normalization (Part
B) is confirmed not to be a viable fix for the resulting magnitude
mismatch, and is a useful cautionary example of trusting raw accuracy on
an imbalanced test set; tuning NMC's solid-state diffusivity at the
simulation source (Part C) produces the first genuine sim-to-real signal
found across this entire investigation -- but only for Random Forest,
and only partially (strong at the top of the target zone, weaker deeper
in; LFP's own mismatch untouched).

Two natural next steps, neither attempted here: (1) investigate why
XGBoost doesn't benefit the way Random Forest does despite seeing the
same improved features -- possibly worth deliberately using Random
Forest as the primary model for this feature set rather than treating
XGBoost's failure as a blocker; (2) a finer-grained or depth-dependent
diffusivity adjustment (rather than one flat factor for the whole
positive electrode) to close the remaining gap in the deepest bins,
and/or the analogous investigation for LFP's own diffusivity parameter
(Prada2013) to address its separate, still-unfixed mismatch.

Full run outputs: `sim_and_real_raw.csv`, `features/ml_features.csv`
(gitignored, regenerable via `simulate_batteries_diffusivity_tuned.py`
then `evaluate_diffusivity_tuned_sim_to_real.py`).
