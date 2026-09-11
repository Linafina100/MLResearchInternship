# Voltage-bin imputation leakage: discovery, mechanism, and fix

## Summary

The LFP/NMC classification pipeline was reporting near-100% accuracy on its
voltage-bin dV/dQ features. That accuracy turned out to be substantially
inflated by a data-leakage artifact in how missing feature values were
imputed. Once fixed, honest accuracy for this feature set is **~78-94%**
depending on pulse-test step count, not ~97-100%.

## The mechanism

`feature_engineering.py`'s `create_features_by_voltage_bins` computes one
dV/dQ feature per 0.1V terminal-voltage bin, spanning 1.9-4.3V. LFP and NMC
have different physical voltage ranges (LFP ~2.0-3.6V, NMC ~2.5-4.2V), so
several bins near either chemistry's ceiling are essentially never reached
by that chemistry — but they *are* reached by the other, so the column
isn't dropped as "entirely empty."

`ml_pipeline.py` then runs `SimpleImputer(strategy='median')` over all
feature columns before training. For a bin where, say, LFP has 0% real
coverage, every LFP row's NaN gets filled with the *same constant* — the
median of NMC's real values in that column. A decision tree can then split
on "does this feature equal that exact constant?" and classify LFP almost
perfectly for free, without having learned anything about real
electrochemistry. This is missing-not-at-random data leakage via median
imputation.

## Evidence: per-chemistry real-data coverage (25-step feature set)

| Bin | LFP coverage | NMC coverage |
|---|---|---|
| dV_dQ_V_3.8_3.7 | 0.0% | 32.5% |
| dV_dQ_V_3.7_3.6 | 0.0% | 64.1% |
| dV_dQ_V_3.6_3.5 | 0.0% | 66.2% |
| dV_dQ_V_3.5_3.4 | 0.0% | 65.8% |
| dV_dQ_V_3.4_3.3 | 80.3% | 36.8% |
| dV_dQ_V_3.3_3.2 | 88.4% | 4.3% |
| dV_dQ_V_3.2_3.1 | 36.1% | 2.6% |

Four bins are 0%-populated for LFP; two more are <5% populated for NMC. Yet
six of these seven bins appeared among the top-5 most important features
across every step count before the fix.

## Before the fix (unfiltered voltage bins)

| Steps | RF accuracy | XGB accuracy |
|---|---|---|
| 5 | 100.0% | 100.0% |
| 10 | 98.9% | 97.9% |
| 15 | 100.0% | 100.0% |
| 20 | 100.0% | 100.0% |
| 25 | 100.0% | 98.96% |

Top-5 features (25-step case, by importance): `dV_dQ_V_3.4_3.3`,
`dV_dQ_V_3.3_3.2`, `dV_dQ_V_3.5_3.4`, `dV_dQ_V_3.7_3.6`, `dV_dQ_V_3.6_3.5` —
four of these five are the structurally one-sided bins from the table above.

## The fix

`create_features_by_voltage_bins` now takes a `min_chemistry_coverage`
parameter (default `0.2`): after the existing "drop entirely-empty" step, it
additionally drops any bin where *either* chemistry's real (non-imputed)
coverage falls below that threshold. See `feature_engineering.py` in this
folder, particularly the block around `min_chemistry_coverage` /
`coverage_by_chem` / `one_sided_cols`.

## After the fix (>20% mutual-coverage bins only)

Only **one** bin survives the filter across all step counts:
`dV_dQ_V_3.4_3.3` (80.3% LFP / 36.8% NMC coverage — the one bin where both
chemistries contribute genuine, varying measurements).

| Steps | RF accuracy | XGB accuracy |
|---|---|---|
| 5 | 78.3% | 78.3% |
| 10 | 91.6% | 90.5% |
| 15 | 90.5% | 90.5% |
| 20 | 93.75% | 93.75% |
| 25 | 89.6% | 89.6% |

25-step classification report (Random Forest, the best model):

```
              precision    recall  f1-score   support

         LFP       1.00      0.80      0.89        50
         NMC       0.82      1.00      0.90        46

    accuracy                           0.90        96
   macro avg       0.91      0.90      0.90        96
weighted avg       0.91      0.90      0.90        96
```

With genuine leakage removed, the model now under-predicts LFP (recall
drops to 80%) whenever the single remaining voltage bin's value is
ambiguous, and over-predicts NMC (recall stays 100%). With only one
predictor left, feature importance is trivially `{dV_dQ_V_3.4_3.3: 1.0}` —
this is no longer a multi-feature classifier, just a single-threshold rule
on the one bin both chemistries honestly share.

## Takeaway

Real, leakage-free accuracy on this feature basis tops out around 90-94%
for 10+ pulse steps, not the ~100% originally reported. The natural next
step is restoring accuracy with *legitimately* shared signal (e.g.
resistance/IR-drop features, or additional bins combined with an explicit
"reached this bin" indicator) rather than relying on one voltage bin alone.
