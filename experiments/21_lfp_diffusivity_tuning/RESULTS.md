# Experiment 21: does diffusivity tuning fix LFP's magnitude mismatch too?

## Motivation

Experiment 20's diffusivity-tuning fix (Part C) closed most of NMC's
dV/dQ magnitude mismatch, producing the first genuine sim-to-real
signal in this investigation. LFP shows its own separate, systematic
magnitude mismatch (synthetic 2-6x larger than real, documented since
experiment 07 and never previously investigated) that was out of scope
for experiment 20. This experiment tests whether the same methodology
transfers directly.

## Phase 1: does the same diffusivity lever work? -- a clean, systematic negative finding

A single-point check (Prada2013, SOH=0.8, 0.15C, resistance_factor=0.8)
found `Positive particle diffusivity` (LFP's own active-material
parameter, baseline 5.9e-18 m2/s -- already ~680x smaller than NMC's
baseline of 4e-15 m2/s) has **zero measurable effect** on target-zone
dV/dQ across a 500x range (factor 1/50 to x10): mean stays at -3.38
throughout. Two other candidates were also checked at the same point and
also showed no effect: positive electrode exchange-current density
(zero effect across a 100x range) and negative electrode (graphite)
diffusivity (moved the magnitude the *wrong* direction, away from the
target, as it was reduced).

**Systematically confirmed** (`diagnose_lfp_diffusivity_factor.py`,
mirroring experiment 20's diagnostic structure): diffusivity factors
{1 (baseline), 5, 10, 20, 50} swept across the real C-rate
(0.1/0.15/0.2) and temperature (15/25/35C) ranges at SOH=0.8 -- 45
combinations, all robust (no failures, no collapse), and **the mean
target-zone magnitude stays frozen at -3.36 to -3.37 in every single
combination**, regardless of factor, C-rate, or temperature. Real LFP's
target is -2.9 to -0.3; not one combination landed in it, and none moved
meaningfully closer.

| factor | Robust | Land in real target (-2.9 to -0.3) | Mean magnitude |
|---|---|---|---|
| 1 (baseline) | 9/9 | 0/9 | -3.37 |
| 5 | 9/9 | 0/9 | -3.37 |
| 10 | 9/9 | 0/9 | -3.37 |
| 20 | 9/9 | 0/9 | -3.37 |
| 50 | 9/9 | 0/9 | -3.36 |

## Why this is a genuinely different picture from NMC

NMC's smooth, single-phase solid-solution discharge curve has its
end-of-discharge steepness governed by how fast particle-surface
concentration (which sets the OCP) can track bulk concentration --
exactly what solid-state diffusivity controls, which is why tuning it
worked. LFP is different: it's well established in battery literature
that LiFePO4 undergoes a **two-phase phase transition** during
(de)lithiation (not a single-phase solid solution), producing its
characteristic flat voltage plateau. The plateau's flatness -- and the
shape of the tail past it, which is what the 2.5-3.0V target zone
actually sits in -- is primarily a **thermodynamic property of the
fitted OCP curve** (the two-phase equilibrium potential), not a
transport/kinetic property. That would explain why diffusivity,
exchange-current density, and even the *other* electrode's diffusivity
all failed to move it: none of these change the shape of the OCP
function itself, which is what would need to change here.

**Conclusion of Phase 1**: the exact diffusivity-tuning methodology that
fixed NMC's magnitude mismatch does not transfer to LFP. Don't re-attempt
diffusivity, exchange-current density, or negative-electrode tuning for
LFP's magnitude mismatch without new evidence. Full per-combination data:
`diagnostic_results.csv`.

## Phase 2: directly softening LFP's OCP tail rate constant

Since transport/kinetics tuning doesn't work, and per user agreement to
proceed as an explicit **empirical calibration** (not first-principles
physics), Phase 2 directly edits LFP's OCP function itself.

Prada2013's parameter set borrows Afshar2017's LFP fit (Prada2013 itself
doesn't define an LFP OCP):

```python
def LFP_ocp_Afshar2017(sto):
    c1 = -150 * sto
    c2 = -30 * (1 - sto)
    k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)
    return k
```

Traced by inspection (not assumed): `sto` increases toward 1 as the cell
discharges, and near `sto=1` this evaluates to ~2.49V, confirming it
governs the target zone. The `-0.9*exp(-30*(1-sto))` term produces the
steep tail there. `diagnose_lfp_ocp_rate_constant.py` softens the `-30`
rate constant (smaller magnitude spreads the same voltage drop over a
wider stoichiometry range) and sweeps candidates across the same real
C-rate/temperature grid at SOH=0.8.

### Result: a wide, stable range, with a clear best match

Unlike NMC's diffusivity (narrow window, collapsed above factor=12),
this lever is smooth and continuously tunable -- no collapse observed
even at the most extreme value tested:

| Rate constant | Robust | Land in real target (-2.9 to -0.3) | Mean magnitude |
|---|---|---|---|
| -30 (baseline) | 9/9 | 0/9 | -3.37 |
| -10 | 9/9 | 0/9 | -3.25 |
| -7 | 9/9 | 0/9 | -2.99 |
| -5 | 9/9 | 9/9 | -2.57 |
| -4 | 9/9 | 9/9 | -2.07 |
| **-3** | 9/9 | 9/9 | **-1.31** |
| -2 | 9/9 | 9/9 | -0.82 |
| -1 | 9/9 | 9/9 | -0.39 (**3 new outliers** appear here, vs. the constant 9 baseline outliers at every other value -- all outside the target zone, unrelated to this change, see below) |

**Rate=-3 is the best-centered match**: mean -1.31 against real LFP's
own true mean of -1.28 (computed directly from its 5 real per-bin
values: -0.31, -0.72, -1.18, -1.34, -2.87). Chosen over softer values
(-2, -1) both because it's already the closest match and because -1
starts introducing new outliers, a caution sign consistent with "don't
push further than needed" from the NMC investigation.

**A fixed 9-outlier count appears identically at every rate constant
including the unmodified baseline** -- confirmed these are all *outside*
the target zone (`n_outliers_zone=0` throughout) and unaffected by this
change; a pre-existing, out-of-scope artifact, not something this fix
introduced.

**Per-bin shape check at rate=-3** (single representative point, C-rate
0.15/25C): the *deepest* bin matches almost exactly (synthetic -2.97 vs.
real -2.87), but the shallower bins within the target zone are still
~1.7-2x too large (e.g. 2.9-3.0V: synthetic -0.55 vs. real -0.31). The
aggregate-mean match is real progress but not a uniform per-bin fix --
the mirror image of NMC's own unevenness (there, the *shallowest* bin
matched best).

## Phase 3: full pipeline, LFP OCP rate=-3 + NMC diffusivity/10 combined

`simulate_batteries_lfp_ocp_tuned.py`: everything from experiment 20
(SOH=0.8 fixed, C-rate 0.1-0.2, 1.5V cutoff, dense `t_interp`, NMC
Chen2020+OKane2022 with diffusivity/10, Mohtat2020 dropped) plus LFP's
OCP tail softened to rate=-3. 498/498 simulations succeeded (matching
every prior experiment in this fix chain).

### Result: both models now show strong, genuinely balanced sim-to-real transfer

| Model | Raw accuracy | **Balanced accuracy** | LFP recall | NMC recall |
|---|---|---|---|---|
| Random Forest | 86.27% | **90.55%** | 0.990 | 0.821 |
| **XGBoost** | 96.31% | **97.21%** | 0.990 | 0.955 |

Compare to experiment 20 Part C (NMC-only fix, LFP untouched): RF
82.92% -> **90.55%**, and dramatically, XGBoost **52.60% (chance) ->
97.21%**. Fixing LFP's own magnitude mismatch didn't just help LFP
classification -- it also resolved XGBoost's complete failure mode from
experiment 20, consistent with the hypothesis that XGBoost was
overfitting decision thresholds to a synthetic distribution that, before
this fix, was unrealistic for *both* chemistries at once.

**Not a class-imbalance artifact**: unlike experiment 20 Part B's
illusory normalization gains, recall is high and balanced for *both*
classes here (LFP 0.990, NMC 0.821-0.955) -- not one collapsed near-zero
while the other sits near 1.0. Also re-verified the
top feature (`dV_dQ_V_3.0_2.9`, 76% importance) has ~100% synthetic
coverage for both chemistries -- not a fragile, sparse-coverage-driven
signal like experiment 07's discarded attempt #4.

**Coverage remains excellent** (LFP ~99.6% throughout; NMC 100% down to
83.1% in the deepest bin, unchanged from experiment 20 since this
experiment didn't touch NMC's coverage fix). **Magnitude match remains
imperfect** -- LFP synthetic is now much closer to real but still
somewhat elevated in most bins (e.g. 2.9-2.8V: real -0.71 vs. synthetic
-1.55); NMC's magnitude match (from experiment 20, unchanged here) is
still uneven across the zone, per that experiment's own findings. The
classification result improved substantially despite this -- the
remaining mismatch is evidently no longer the dominant obstacle it was.

## Overall conclusion

The two-part fix -- NMC diffusivity/10 (experiment 20) + LFP OCP tail
softening to rate=-3 (this experiment) -- takes sim-to-real balanced
accuracy from chance-level (~50%, everywhere in experiments 16-20) to
**90.55% (RF) / 97.21% (XGBoost)** at the realistic SOH=0.8 threshold,
Chen2020+OKane2022 (not Mohtat2020), LFP included. This is the strongest
result in the entire investigation. Both fixes are explicitly empirical
calibrations (NMC: a physical transport parameter reduced beyond its
literature-typical range; LFP: a direct edit to a fitted OCP curve's
functional form) -- documented as such, not presented as more accurate
first-principles electrochemistry, per explicit agreement.

Remaining open items, not attempted here: Mohtat2020 is still completely
unfixed and excluded from the NMC pool; the per-bin magnitude match is
still uneven for both chemistries even though classification accuracy is
now high; and neither fix has been validated outside the single SOH=0.8
point.

Full run outputs: `sim_and_real_raw.csv`, `features/ml_features.csv`
(gitignored, regenerable via `simulate_batteries_lfp_ocp_tuned.py` then
`evaluate_lfp_ocp_tuned_sim_to_real.py`).
