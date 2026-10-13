# optimize_empa_low_soc, attempt 03: tuned negative-electrode factor

## The idea

Attempt 02's `NEG_CAPACITY_FACTOR=0.7` was validated for a *different*
campaign's baseline (fixed SOH=0.8, narrower real C-rate selection),
not tuned against this pipeline's own SOH-range synthetic data or this
pipeline's own (much broader, `MAX_CYCLES_PER_CELL=600`) real sample.
It closed much of the LFP calibration gap but not all of it. Swept the
factor against this pipeline's own data before assuming 0.7 was optimal
here.

## Sweep (small scale, 20 variations/size, before any full run)

Total absolute deviation from real LFP's median dV/dQ across the four
most populated real-LFP bins (3.1-3.0V, 3.0-2.9V, 2.9-2.8V, 2.8-2.7V):

| Factor | 3.1-3.0V | 3.0-2.9V | 2.9-2.8V | 2.8-2.7V | Total abs. error |
|---|---|---|---|---|---|
| 0.4 | -1.00 | -3.68 | -4.13 | -5.67 | 3.13 |
| 0.42 | -0.93 | -3.50 | -- | -5.35 | 2.53 |
| 0.45 | -0.86 | -3.25 | -- | -4.90 | 1.99 |
| **0.47** | **-0.81** | **-3.06** | **--** | **-4.67** | **1.88** |
| 0.5 | -0.76 | -2.87 | -- | -4.25 | 2.35 |
| 0.55 | -0.66 | -2.49 | -- | -3.77 | 3.14 |
| 0.6 | -0.62 | -2.13 | -2.81 | -3.33 | 4.21 |
| 0.7 (attempt 02) | -0.53 | -1.52 | -2.44 | -2.67 | 5.93 |
| 0.8 | -0.49 | -0.95 | -2.17 | -2.20 | 7.27 |
| Real LFP (target) | -1.87 | -2.65 | -3.50 | -5.07 | -- |

Clear minimum around 0.45-0.47 -- substantially better fit than 0.7
across every bin. Picked **0.45** (clean round number near the
empirical optimum) for the full run rather than over-fitting to a
small-sample (20 variations/size) diagnostic -- consistent with this
project's standing lesson to validate against the real classification
evaluation, not just a diagnostic metric.

## Results (full scale: 498 synthetic batteries, same 151,074 real cycles reused from attempt 01)

| Initial_SOC | RF balanced (02 -> 03) | RF LFP recall (02 -> 03) | XGB balanced (02 -> 03) |
|---|---|---|---|
| 1.0 | 99.87% -> 99.87% | 1.000 -> 1.000 | 99.87% -> 99.87% |
| 0.9 | 99.87% -> 99.87% | 1.000 -> 1.000 | 99.87% -> 99.87% |
| 0.8 | 49.87% -> **63.93%** | **0.000 -> 0.281** | 99.87% -> 99.87% |
| 0.7 | 49.87% -> 49.87% | 0.000 -> 0.000 | 99.88% -> 99.87% |
| 0.6 | 49.87% -> 49.89% | 0.000 -> 0.000 | 99.89% -> 99.89% |
| 0.5 | 49.87% -> 49.90% | 0.000 -> 0.000 | 99.91% -> 99.90% |
| 0.4 | 49.93% -> 49.92% | 0.000 -> 0.000 | 99.93% -> 99.91% |
| 0.3 | 49.99% -> 49.99% | 0.000 -> 0.000 | 99.99% -> 99.99% |
| 0.2 | 49.99% -> 49.99% | 0.000 -> 0.000 | 99.99% -> 99.99% |
| 0.15 | 49.99% -> 49.99% | 0.000 -> 0.000 | 99.99% -> 99.99% |
| 0.1 | 49.99% -> 50.00% | 0.000 -> 0.000 | 50.00% -> 49.99% |
| 0.05 | 50.00% -> 50.00% | 0.000 -> 0.000 | 50.00% -> 50.00% |

(Peak RSS: 2.86GB -- consistent with prior attempts, no memory concern.)

## Interpretation: diminishing returns from this lever alone

The tighter LFP calibration gives **one genuine but partial win**:
Random Forest's recall at Initial_SOC=0.8 goes from pure chance
(LFP_recall=0.000) to partial signal (0.281) -- the first sign of life
RF has shown anywhere outside 1.0/0.9 across all three attempts.

**Everywhere else, essentially no change.** XGBoost was already
saturated (99.87-99.99%) across 1.0-0.15 with the looser 0.7
calibration -- there was no headroom left for a tighter fit to exploit
in that range, and it doesn't help below 0.15 either (still chance at
0.1/0.05 in both attempts). Random Forest below 0.7 and XGBoost below
0.15 are apparently bottlenecked by something this lever alone doesn't
reach.

**This suggests the negative-electrode-balance fix, on its own, is
approaching its ceiling for this specific calibration gap.** Further
gains likely need a different/additional lever, not further tuning of
this same factor.

## Files

- `data_generation/simulate_batteries.py` -- attempt 02's generator with `NEG_CAPACITY_FACTOR=0.45`.
- `data_generation/build_real_empa.py` -- unmodified copy, NOT invoked (real data reused from attempt 01).
- `run_evaluation.py` -- unmodified from attempts 01/02.
- `outputs/sweep_results.json` -- this attempt's results.

## Next steps

The other validated ingredient from `optimize_empa/experiment/01_ratio_to_median_baseline/`
-- **ratio-to-median scale-invariant features** (each bin's dV/dQ
divided by that sample's own median across its own observed bins,
instead of absolute dV/dQ) -- was specifically designed for exactly
this situation: classification at low SOC where only a narrow,
possibly offset voltage window is observed. It's independent of (and
stacks with) the negative-electrode physics fix; worth trying next,
particularly to see whether it's what's needed to push past XGBoost's
0.15 floor and recover Random Forest below 0.7.
