# Experiment 23: Initial_SOC sweep for EMPA real test data

## Motivation

Experiment 22 (merged to `main`) tested the calibrated sim-to-real
pipeline (NMC diffusivity/10, LFP OCP rate=-3) against real EMPA
discharge cycles, but every real cycle used was a **full** discharge,
starting from a full charge (~100% SOC) down to the protocol's cutoff.
The synthetic training data itself already randomizes `Initial_SOC`
(checked directly: spread roughly uniformly across ~0.50-0.99, not fixed
at 1.0), so the model was already trained on partial-discharge starts --
but that had never been tested against **real** partial-discharge
windows. This experiment does that, extending the sweep idea from the
pre-dating, fully-synthetic `experiments/06_cont_random_discharge_soc_sweep/`
to the real EMPA data.

## Method

New branch `experiment-23-empa-soc-sweep`, no resimulation -- reuses
`data/soh_0.8_lfp_ocp_tuned_v1.5/` as training data, exactly as
experiments 21/22 did.

`build_real_empa_soc_sweep_dataset.py` reuses experiment 22's cycle
selection unchanged (same chemistry/SOH/C-rate logic, same rest-phase
and rescale-to-synthetic-scale fixes for the three data-quality bugs
documented in `experiments/22_empa_rocrate_sim_to_real/RESULTS.md`). For
every real cycle at SOH=0.8+/-0.02, it derives 8 truncated sub-traces --
one per `Initial_SOC` in `[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]` --
each keeping only the portion of that same physical discharge curve
occurring after the cell has already delivered `(1-s)` of that cycle's
own realized capacity. Time and Capacity are reset to 0 at the new start.
`s=1.0` reproduces the exact same real cycles experiment 22 tested
against. `0.4` and `0.3` deliberately push below the synthetic training
data's own Initial_SOC range (~0.50-0.99) to test extrapolation, not just
interpolation. A truncated variant is discarded whenever it would clip
past the 2.5-3.0V target zone (checked directly against that cycle's own
real voltage trace) -- in practice, this never triggered: even at
`s=0.3`, every one of the 10,993 matched cycles still reaches into the
target zone, because that zone sits deep in the discharge tail and
truncation only removes the *shallow* portion. Result: 87,944 real
SOC-sweep variants (10,993 cycles x 8 buckets), all buckets equally
sized.

`evaluate_empa_soc_sweep.py` trains one model (same hyperparameters as
experiments 20-22) on the full synthetic set, then evaluates it
separately against each `Initial_SOC` bucket, reporting balanced accuracy
and per-class recall per bucket.

## A feature-set difference from experiment 22, and why it's not a bug

The `s=1.0` bucket is, by construction, the exact same 10,993 real cycles
experiment 22 tested against -- intended as a correctness check that
should reproduce its published 97.89% RF / 92.71% XGBoost balanced
accuracy. Random Forest matched almost exactly (97.89%), but XGBoost did
not (97.90% here, not 92.71%).

Root cause, confirmed by inspection: `feature_engineering.py`'s
mutual-coverage filter (>=20% real+synthetic coverage per bin, else the
bin is dropped) runs on the **combined, pooled** real+synthetic dataset.
In experiment 22, only the `s=1.0` real data existed, and NMC's coverage
in the deepest bin (`dV_dQ_V_2.6_2.5`) was just 2% there -- below the 20%
threshold, so that bin was dropped, leaving 4 target-zone bins. Here, all
8 `Initial_SOC` buckets are pooled together for that same filter, and
lower-SOC buckets have much higher coverage in that bin by construction
(truncating removes the shallow portion of the curve, so a lower-SOC
window sits closer to the cutoff and is more likely to reach the deepest
bin -- observed directly: NMC coverage in `dV_dQ_V_2.6_2.5` rises from 2%
at `s=1.0` to 54% at `s=0.3`). Pooled together, that bin clears 20%
coverage and survives, giving **5** target-zone bins here instead of 4.
The model trained on this run therefore has one more (evidently useful)
feature than experiment 22's did -- not a different or buggy pipeline,
just a different, data-dependent feature-selection outcome from running
the same filter on a broader pooled real set. This is worth being
explicit about rather than silently reporting a "reproduction" that
wasn't quite exact.

## Result

| Initial_SOC | n | RF balanced | RF LFP recall | RF NMC recall | XGB balanced | XGB LFP recall | XGB NMC recall |
|---|---|---|---|---|---|---|---|
| 1.0 | 10,993 | 97.89% | 1.000 | 0.958 | 97.90% | 1.000 | 0.958 |
| 0.9 | 10,993 | 98.50% | 1.000 | 0.970 | 98.54% | 1.000 | 0.971 |
| 0.8 | 10,993 | 97.93% | 1.000 | 0.959 | 98.03% | 1.000 | 0.961 |
| 0.7 | 10,993 | 98.10% | 0.995 | 0.967 | 98.21% | 0.993 | 0.971 |
| 0.6 | 10,993 | 98.23% | 0.995 | 0.970 | 98.29% | 0.995 | 0.971 |
| 0.5 | 10,993 | 98.46% | 0.998 | 0.971 | 98.49% | 0.998 | 0.972 |
| 0.4* | 10,993 | 98.53% | 0.999 | 0.972 | 98.63% | 0.999 | 0.974 |
| 0.3* | 10,993 | 98.28% | 0.993 | 0.973 | 98.32% | 0.992 | 0.974 |

\* extrapolation: below the synthetic training data's own Initial_SOC
range (~0.50-0.99).

**Balanced accuracy never drops below 97.89% anywhere in the sweep**,
including both extrapolation buckets. If anything, the trend is mildly
*upward* as Initial_SOC decreases -- consistent with the coverage
explanation above (lower-SOC windows sit closer to the discharge tail, so
the deep target-zone bins are reached more consistently, giving the
model more complete real feature vectors to classify from). This should
not be over-read as "the model gets better at lower SOC" in a deep sense
-- it's a direct consequence of how truncation interacts with real-data
coverage, not independent evidence about chemistry-signal strength at
low SOC.

## Conclusion

**The sim-to-real result is stable across the full Initial_SOC sweep,
including genuine extrapolation beyond the synthetic training
distribution.** Real partial-discharge windows -- even ones starting from
just 30% of a cycle's own realized capacity, well outside anything the
synthetic training data covered -- classify just as well as, or slightly
better than, full real discharges. Combined with experiment 22's finding
that the calibration generalizes to an independent real dataset, this is
further evidence the underlying dV/dQ target-zone signal (dominated by
`dV_dQ_V_3.0_2.9` and `dV_dQ_V_2.9_2.8`, as in every prior experiment) is
a genuine, robust chemistry-distinguishing feature -- not a fragile
artifact of matching one specific real discharge-starting condition.

**Caveats, stated plainly:**
- The `s=1.0` bucket does not exactly reproduce experiment 22's published
  XGBoost number, for the feature-set reason explained above -- not a
  pipeline bug, but a real methodological difference from pooling a wider
  real dataset into the same coverage filter. Random Forest's near-exact
  match suggests it is more robust to this specific feature-count
  difference; note RF has been the more consistently reliable model
  across experiments 20-23.
- All 8 buckets in a given row come from the **same 10,993 underlying
  physical cycles**, just windowed differently -- they are not 8
  independent real test sets. The consistently high accuracy across
  buckets says the *classification signal in the target zone survives
  truncation*; it does not multiply the effective amount of independent
  real evidence behind these results the way experiment 22's larger,
  genuinely-distinct real dataset did.
- The truncation guard (discard if it clips past the target zone) never
  actually triggered for any cycle at any tested SOC level down to 0.3 --
  so this sweep does not establish where such a guard *would* start
  mattering; it simply confirms it wasn't needed in this range.
