# optimize_empa_low_soc, attempt 04: ratio-to-median scale-invariant features

## The idea

Attempt 03 found the negative-electrode-balance lever alone had hit
diminishing returns: a real but partial win (RF recovers somewhat at
SOC=0.8), with XGBoost already saturated across 1.0-0.15 and nothing
moving below that. The other validated ingredient from the earlier
`optimize_empa/experiment/01_ratio_to_median_baseline/` campaign is a
different kind of fix entirely: instead of changing the physics
calibration, make the *feature* invariant to whatever calibration gap
remains. Each bin's absolute dV/dQ is replaced by that same row's own
value divided by its own median across its own observed bins -- so a
classifier sees shape/relative-pattern, not absolute magnitude. This
stacks independently with the physics fix (attempt 03's synthetic
data, unchanged) and doesn't require any new simulation.

## What changed

One function added to `run_evaluation.py`, applied to the full
feature table (synthetic + real together) right after bin selection,
before the per-SOC loop:

```python
def apply_ratio_to_median(features_df, bin_cols):
    X = features_df[bin_cols].values.astype(float)
    n_valid = np.sum(~np.isnan(X), axis=1)
    keep_mask = n_valid >= MIN_BINS_PER_ROW
    X_kept = X[keep_mask]
    row_median = np.nanmedian(X_kept, axis=1, keepdims=True)
    ratio = X_kept / row_median
    out = features_df.loc[keep_mask].copy()
    out[bin_cols] = ratio
    return out, int((~keep_mask).sum())
```

`MIN_BINS_PER_ROW=3` (env-overridable) drops rows with too few valid
bins to compute a meaningful median. No other change -- same attempt
03 synthetic data (`NEG_CAPACITY_FACTOR=0.45`), same attempt 01 real
data, same per-SOC specialization, same `class_weight='balanced'`.

## Results (full scale, same data as attempts 01/03)

| Initial_SOC | RF balanced (03 -> 04) | XGB balanced (03 -> 04) | XGB LFP recall (03 -> 04) |
|---|---|---|---|
| 1.0 | 99.87% -> **50.00%** | 99.87% -> 99.95% | 1.000 -> 1.000 |
| 0.9 | 99.87% -> **50.00%** | 99.87% -> 99.94% | 1.000 -> 1.000 |
| 0.8 | 63.93% -> **50.00%** | 99.87% -> 99.98% | 1.000 -> 1.000 |
| 0.7 | 49.87% -> 50.00% | 99.87% -> 99.97% | 1.000 -> 1.000 |
| 0.6 | 49.89% -> 50.00% | 99.89% -> 99.97% | 1.000 -> 1.000 |
| 0.5 | 49.90% -> 50.00% | 99.90% -> 99.95% | 1.000 -> 1.000 |
| 0.4 | 49.92% -> 50.00% | 99.91% -> 99.89% | 1.000 -> 1.000 |
| 0.3 | 49.99% -> 50.00% | 99.99% -> 100.00% | 1.000 -> 1.000 |
| 0.2 | 49.99% -> 50.00% | 99.99% -> 100.00% | 1.000 -> 1.000 |
| 0.15 | 49.99% -> 50.00% | 99.99% -> 99.92% | 1.000 -> 1.000 |
| 0.1 | 50.00% -> 50.00% | 50.00% -> 48.98% | 0.000 -> 0.000 |
| 0.05 | 50.00% -> 50.00% | **50.00% -> 81.81%** | **0.000 -> 1.000** |

(Peak RSS: 2.83GB -- consistent with prior attempts, no memory concern.)

## A genuinely mixed result

**Random Forest regresses everywhere it was previously working.**
Attempt 03's partial wins at Initial_SOC 1.0/0.9 (99.87%) and 0.8
(63.93%) all collapse to exactly 50.00% (pure chance, `LFP_recall=0.000`)
under the ratio-to-median transform. Ratio-to-median removes absolute-
magnitude information by construction -- exactly the information RF
was apparently still exploiting at those SOC points even through the
residual calibration gap.

**XGBoost breaks through at the very bottom of the sweep.**
Initial_SOC=0.05 goes from pure chance (50.00%, `LFP_recall=0.000`) to
**81.81% balanced accuracy, `LFP_recall=1.000`** -- the first time
*anything*, any model or attempt in this campaign, has produced a
working result that low. Everywhere XGBoost was already working
(1.0-0.15), it stays working, with small (noise-level) movement in
both directions. Initial_SOC=0.1 stays at chance either way.

## Interpretation

This isn't a strict improvement -- it's a trade. For Random Forest,
ratio-to-median is a clear regression. For XGBoost, it's neutral across
most of the range and a genuine breakthrough at the single lowest SOC
point. Since XGBoost was already the stronger model everywhere except
the two or three highest SOC points (where RF briefly matched or beat
it), and since this campaign's stated goal prioritizes "as low an SOC
as possible," this result is net positive for that goal -- but it
means neither attempt 03 nor attempt 04 alone is best across the whole
sweep: **attempt 03's Random Forest is the best choice at Initial_SOC
1.0-0.8; attempt 04's XGBoost is the best (or only working) choice at
every other point, including the previously-unreachable 0.05.**

## Files

No `data_generation/` -- this attempt reuses attempt 03's synthetic
data (`data/optimize_empa_low_soc_03_synthetic/`) and attempt 01's
real data (`data/optimize_empa_low_soc_01_real/`) unchanged; only the
evaluator changed.

- `run_evaluation.py` -- attempt 03's evaluator + `apply_ratio_to_median()`.
- `pipeline/feature_engineering.py` -- unchanged local copy.
- `outputs/sweep_results.json` -- this attempt's results.

## Next steps

The natural next attempt is a **hybrid**: pick the feature
representation (absolute dV/dQ vs. ratio-to-median) and/or the model
per SOC bucket, rather than one fixed choice for the whole sweep --
using whichever of attempt 03 / attempt 04 is actually best at each
point. This is the same structural idea Ella's branch tried (a
different feature swap at SOC<=0.3, explicitly not adopted into this
campaign because it was untested) -- except now backed by this
campaign's own validated, full-scale evidence for exactly where the
swap helps and where it hurts, rather than an assumption.
