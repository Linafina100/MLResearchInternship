# optimize_empa_low_soc, attempt 02: negative-electrode capacity rebalance

## The idea

Attempt 01 root-caused its Random-Forest collapse (and XGBoost's
collapse below Initial_SOC 0.15) to a genuine calibration gap: real
LFP's dV/dQ magnitude in the 2.7-3.3V range it actually populates sits
much closer to synthetic NMC's calibration than to synthetic LFP's own
-- the synthetic LFP OCP model (Afshar2017-based, tail rate constant
-3) simply doesn't dive as steeply as real EMPA LFP cells do in that
range.

A separate, earlier campaign against the same real EMPA dataset
(`optimize_empa/experiment/01_ratio_to_median_baseline/`) found and
validated a fix for exactly this kind of gap: scaling LFP's
**negative-electrode** maximum/initial concentration by a factor
independent of the positive electrode's own SOH scaling shifts *where*
the full-cell voltage dive happens (a capacity-balance effect, not a
change to the OCP curve itself) without touching the OCP tail
parameters (already known to be near their effective ceiling -- see
project memory `exp29_lfp_ocp_tail_ceiling`). That campaign validated
`NEG_CAPACITY_FACTOR=0.7` end-to-end, not just against a narrow
diagnostic.

## What changed

One line added to `data_generation/simulate_batteries.py`'s `solve_job()`,
applied to LFP only, right after the existing SOH scaling:

```python
if chem == "LFP":
    param["Maximum concentration in negative electrode [mol.m-3]"] *= NEG_CAPACITY_FACTOR
    param["Initial concentration in negative electrode [mol.m-3]"] *= NEG_CAPACITY_FACTOR
```

`NEG_CAPACITY_FACTOR` defaults to 0.7 (env-overridable). Nothing else
changed -- same job generation, same evaluator, same real data (reused
directly from attempt 01, since real-data generation doesn't depend on
this fix).

## Sanity check before the full run

Compared median dV/dQ per bin between attempt 01 (unfixed) and this
fix, at a small scale (20 variations/size) before committing to the
full 498-battery run:

| Bin | Attempt 01 LFP (unfixed) | Attempt 02 LFP (fixed) | Real LFP (for reference) |
|---|---|---|---|
| 3.1-3.0V | -0.44 | -0.53 | -1.87 |
| 3.0-2.9V | -0.50 | **-1.52** | -2.65 |
| 2.9-2.8V | -1.24 | **-2.44** | -3.50 |
| 2.8-2.7V | -1.82 | **-2.67** | -5.07 |

Substantial movement toward real LFP's calibration at every one of
these bins, closing much of (though not all of) the gap -- worth
running the full evaluation.

## Results (full scale: 498 synthetic batteries, same 151,074 real cycles as attempt 01)

| Initial_SOC | RF balanced (01 -> 02) | RF LFP recall | XGB balanced (01 -> 02) | XGB LFP recall |
|---|---|---|---|---|
| 1.0 | 49.87% -> **99.87%** | 1.000 | 99.89% -> 99.87% | 1.000 |
| 0.9 | 49.87% -> **99.87%** | 1.000 | 99.89% -> 99.87% | 1.000 |
| 0.8 | 49.87% -> 49.87% | 0.000 | 99.87% -> 99.87% | 1.000 |
| 0.7 | 49.87% -> 49.87% | 0.000 | 95.72% -> 99.88% | 1.000 |
| 0.6 | 49.87% -> 49.87% | 0.000 | 99.89% -> 99.89% | 1.000 |
| 0.5 | 49.87% -> 49.87% | 0.000 | 99.91% -> 99.91% | 1.000 |
| 0.4 | 49.93% -> 49.93% | 0.000 | 99.95% -> 99.93% | 1.000 |
| 0.3 | 49.99% -> 49.99% | 0.000 | 99.99% -> 99.99% | 1.000 |
| 0.2 | 49.99% -> 49.99% | 0.000 | 98.43% -> **99.99%** | 1.000 |
| 0.15 | 49.99% -> 49.99% | 0.000 | **49.99% -> 99.99%** | **0.000 -> 1.000** |
| 0.1 | 49.89% -> 49.99% | 0.000 | 50.00% -> 50.00% | 0.000 |
| 0.05 | 50.00% -> 50.00% | 0.000 | 50.00% -> 50.00% | 0.000 |

**Real, meaningful improvement:**
- **XGBoost's working range extended a full SOC step lower**, from
  "works down to 0.2" to **"works down to 0.15"** (49.99% -> 99.99%
  balanced accuracy, LFP recall 0.000 -> 1.000).
- **Random Forest recovered at the two highest SOC points** (1.0, 0.9:
  49.87% -> 99.87%), though it remains collapsed everywhere from 0.8
  down.

**Still unresolved:**
- RF stays at chance for Initial_SOC 0.8 and below.
- Both models still collapse to chance at 0.1 and 0.05 -- the fix
  narrowed but didn't eliminate the calibration gap; what's left is
  apparently enough to still break classification at the very bottom
  of the sweep.

(Peak RSS during evaluation: 2.67GB -- consistent with attempt 01, no
memory concern.)

## Interpretation

`NEG_CAPACITY_FACTOR=0.7` was validated in a *different* campaign
against a *different* synthetic baseline (no SOH-range sampling, fixed
SOH=0.8, different NMC diffusivity/LFP OCP tuning history). That it
transfers here at all -- closing a meaningful chunk of the same kind of
gap, on a pipeline with SOH sampled across 0.8-1.0 and a much broader
real C-rate selection (0.05-1.0 vs. the other campaign's narrower
range) -- is a good sign the underlying physics lever is real and not
narrowly overfit to one calibration. That it doesn't fully close the
gap (RF still collapses below 1.0-0.9; both models still collapse below
0.15) suggests 0.7 may not be this pipeline's own optimal value -- it
was tuned for a different baseline.

## Files

- `data_generation/simulate_batteries.py` -- attempt 01's generator + the one-line fix above.
- `data_generation/build_real_empa.py` -- unmodified copy, NOT invoked (real data reused from attempt 01).
- `run_evaluation.py` -- unmodified from attempt 01 (reads attempt 02's synthetic data, attempt 01's real data).
- `outputs/sweep_results.json` -- this attempt's results.

## Next steps

Sweep `NEG_CAPACITY_FACTOR` itself (e.g. 0.5, 0.6, 0.8) against this
pipeline's own data, rather than assuming 0.7 is optimal here -- the
partial-but-incomplete closure of the calibration gap suggests there's
room to tune further, with a priority on recovering Random Forest
below Initial_SOC=0.9 and pushing both models' working range below
0.15.
