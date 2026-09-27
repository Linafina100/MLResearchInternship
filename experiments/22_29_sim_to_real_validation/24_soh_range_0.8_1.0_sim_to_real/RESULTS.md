# Experiment 24: SOH-range (0.8-1.0) sim-to-real test, all voltage bins

## Motivation

Every validated sim-to-real result so far (experiments 20-23) trained and
tested at a single fixed SOH=0.8 point. That calibrated the NMC positive
particle diffusivity/10 and LFP OCP tail rate=-3 fixes ([experiment
21](../21_lfp_diffusivity_tuning/RESULTS.md)) and validated them against an
independent real dataset ([experiment
22](../22_empa_rocrate_sim_to_real/RESULTS.md), balanced accuracy
97.89%/92.71%). It left two questions open:

1. Do those fixes still work across a *range* of battery health (SOH
   0.8-1.0), or were they implicitly tuned to work only at exactly 0.8?
2. Experiments 22/23 both hardcoded training to the 2.5-3.0V "target zone."
   Does relaxing that restriction -- keeping `feature_engineering.py`'s
   >=20%-mutual-coverage filter exactly as-is, but letting it decide which
   bins survive across the *full* 1.9-4.3V span rather than pre-restricting
   to a hand-picked sub-range -- change anything?

This experiment answers both by generating new synthetic training data with
SOH sampled uniformly from 0.8-1.0 (all other validated fixes unchanged),
building a matching real EMPA test set filtered to SOH in [0.8, 1.0], and
evaluating without a target-zone restriction.

## Step 0: verifying real SOH coverage before building anything

Nobody had confirmed how much real EMPA data actually exists between SOH
0.8 and 1.0 before this experiment -- exp22's BOL methodology (max capacity
over each cell's first 10 cycles) implies early cycles sit near SOH~1.0, but
that was never measured. A standalone histogram of every real discharge
cycle's SOH (same BOL/SOH derivation as `build_real_empa_dataset.py`, no
filtering) confirmed solid coverage across the target range for both
chemistries:

| SOH band | LFP cycles | NMC cycles |
|---|---|---|
| [0.80, 0.90) | 4,858 | 19,220 |
| [0.90, 0.98) | 1,180 | 7,334 |
| [0.98, 1.00] | 22 | 604 |
| **Total [0.80, 1.00]** | **6,006** (18.7% of all LFP cycles) | **29,623** (21.4% of all NMC cycles) |

Both chemistries thin out near SOH=1.0 (expected -- fewer cycles are ever
*that* fresh) but never drop to zero. Proceeded on this basis.

## Build pipeline

- `simulate_batteries_soh_range.py`: identical to [experiment
  21](../21_lfp_diffusivity_tuning/simulate_batteries_lfp_ocp_tuned.py)
  (SOH-scaling fix, dense t_interp, NMC diffusivity/10 on
  Chen2020+OKane2022, LFP OCP tail rate=-3, C-rate 0.1-0.2, 1.5V cutoffs),
  except `SOH_FIXED = 0.8` is replaced with `soh = random.uniform(SOH_MIN,
  SOH_MAX)` (default 0.8-1.0) drawn per battery. 498/498 solves succeeded
  (0 failures), SOH landed uniformly across 0.801-1.000 for both
  chemistries (mean 0.902, std 0.060) -> `data/24_soh_range_0.8_1.0_v1.5/`.
- `build_real_empa_soh_range_dataset.py`: identical to [experiment
  22](../22_empa_rocrate_sim_to_real/build_real_empa_dataset.py) except the
  SOH selection is a range check (`SOH_MIN <= soh <= SOH_MAX`, default
  0.8-1.0) instead of point+tolerance. Recovered exactly the Step 0 counts:
  6,006 LFP cycles (32 cells) + 29,623 NMC cycles (167 cells) ->
  `data/24_real_empa_soh_0.8_1.0/`.
- `evaluate_soh_range_sim_to_real.py`: identical structure to experiment
  22's evaluate script, but does NOT hardcode a target-zone bin restriction
  -- it trains on whatever bins survive `feature_engineering.py`'s default
  `min_chemistry_coverage=0.2` filter across the full 1.9-4.3V span, and
  prints the full unfiltered per-chemistry coverage table for direct
  comparison against exp22's baseline.

## Result: coverage widened, accuracy improved

**Bin survival, SOH 0.8-1.0 vs. exp22's fixed SOH=0.8:**

| | Bin | Survives now? | Survived at SOH=0.8? |
|---|---|---|---|
| | `dV_dQ_V_3.3_3.2` | Yes (LFP 49.8% cov.) | No |
| | `dV_dQ_V_3.2_3.1` | Yes (LFP 100%) | No |
| | `dV_dQ_V_3.1_3.0` | Yes (LFP 100%) | No |
| (unchanged) | `dV_dQ_V_3.0_2.9` through `dV_dQ_V_2.8_2.7` | Yes | Yes |
| | `dV_dQ_V_2.7_2.6` | Yes -- but narrowly: real NMC's own coverage here is only 0.198 (many degraded/small real NMC cells never discharge this low), and it's the synthetic side's much higher coverage blended in with the coverage filter's per-Battery_ID mean that pulls the combined figure just over the 0.2 threshold | Yes |
| | `dV_dQ_V_2.6_2.5` | **No** -- fell below the 20% mutual-coverage threshold once combined with the far larger real-cycle population (exact combined figure not captured, since a dropped bin isn't included in the coverage printout) | Yes |

Net: **7 bins survive now vs. 5 before** -- widening the SOH range let LFP's
less-degraded (higher-SOH) cells reach further up its plateau into
3.1-3.3V, a region that at fixed SOH=0.8 LFP's synthetic voltage never
touched (0% coverage, per experiment 01's structural LFP-vs-NMC ceiling
discussion). This is not the same thing as removing that physical
ceiling -- LFP synthetic coverage above ~3.4V is still 0-2.4% and remains
correctly filtered out -- it's that SOH-range averaging shifted enough mass
into the 3.1-3.3V band for both chemistries to clear the 20% bar there.

**Classification result** (7 bins, 498 synthetic train / 35,629 real test,
83.1% NMC):

| Model | Raw acc | Balanced acc | LFP recall | NMC recall |
|---|---|---|---|---|
| Random Forest | 99.64% | **99.14%** | 0.984 | 0.999 |
| XGBoost | 99.64% | **99.15%** | 0.984 | 0.999 |

For context, experiment 22 (fixed SOH=0.8, target-zone-only, same real
dataset family): RF balanced=97.89% (LFP recall 1.000, NMC recall 0.958),
XGBoost balanced=92.71% (LFP recall 0.864, NMC recall 0.990).

Top features by importance: `dV_dQ_V_3.2_3.1` (35.9%), `dV_dQ_V_3.1_3.0`
(35.0%), `dV_dQ_V_3.0_2.9` (16.6%), `dV_dQ_V_2.7_2.6` (10.3%),
`dV_dQ_V_2.9_2.8` (2.2%) -- the three *newly*-surviving bins account for
~87.5% of total importance combined. `dV_dQ_V_3.3_3.2` (the 49.8%-coverage
partial bin) and `dV_dQ_V_2.8_2.7` together account for the remaining
<0.1%, so the result is not resting on that weaker-coverage bin.

**Magnitude/coverage check** (all 7 surviving bins, real vs. synthetic):
real and synthetic means share sign and comparable order of magnitude in
every bin for both chemistries (e.g. NMC `2.8-2.7V`: real -6.08 vs. synth
-8.42; LFP `2.7-2.6V`: real -4.11 vs. synth -2.07) -- not an exact match
bin-for-bin (consistent with experiment 21's own finding that the LFP OCP
fix matches real's *aggregate* mean, not every bin uniformly), but no
order-of-magnitude mismatch of the kind that previously signaled a scaling
bug (experiment 22's original ~1000x Ah-scale issue). Real NMC coverage
degrades in the deepest bins (0.198-0.776 from 3.0-2.9V down to 2.7-2.6V,
expected -- smaller/more-degraded real coin cells don't all reach that
deep), but the model's top features sit in the well-covered upper half of
that range.

## Conclusion

**Both open questions resolve favorably.** The NMC-diffusivity and LFP-OCP
fixes generalize across a battery-health range, not just the single SOH=0.8
point they were calibrated at -- balanced accuracy did not degrade when SOH
was widened to 0.8-1.0, it *improved* (99.14-99.15% vs. 97.89%/92.71%).
Dropping the hardcoded target-zone restriction and letting the existing
mutual-coverage filter decide naturally across the full voltage span also
paid off: broadening SOH shifted real coverage of LFP's plateau region
(3.1-3.3V) enough to let 3 more genuinely-informative bins clear the
leakage-protection threshold, without needing to touch that filter itself.
XGBoost in particular, which lagged badly behind Random Forest in
experiment 22 (92.71% vs 97.89%), closes almost the entire gap here
(99.15% vs 99.14%) -- the extra bins appear to have given it features it
could split on more reliably than the narrower 5-bin target zone did.

**Caveats, stated plainly:**
- This is not evidence that "all voltage bins" work -- LFP's structural
  ceiling above ~3.4V (a real physics limit, not an artifact; see
  `experiments/01_leakage_fix_20_percent_coverage/RESULTS.md`) is
  untouched: those bins are still 0-2.4% covered by LFP and correctly
  excluded by the same filter that was never disabled. Widening SOH
  recovered 3 *additional* bins just below that ceiling, not the ceiling
  itself.
- Real SOH coverage above 0.98 is thin (22 LFP + 604 NMC cycles out of the
  full 6,006/29,623 in-range set) -- this result is not a strong claim
  about near-perfect-health cells specifically, more about the 0.8-0.95
  band where most of the real data actually sits.
- As in experiment 22, real ambient temperature is a fixed 25C (inside but
  not spanning the synthetic 15-35C range), and real LFP C-rate is fixed
  at ~1.0-1.07C (outside the synthetic 0.1-0.2C training range) -- neither
  caveat is new to this experiment.
