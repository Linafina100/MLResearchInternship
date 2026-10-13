# Voltage-bin imputation leakage: discovery, mechanism, and fix

## Summary

The LFP/NMC classification pipeline was reporting near-100% accuracy on its
voltage-bin dV/dQ features. That accuracy turned out to be substantially
inflated by a data-leakage artifact in how missing feature values were
imputed. Once fixed, honest accuracy for this feature set is **~78-94%**
depending on pulse-test step count, not ~97-100%.

## The mechanism

(This section describes the pulse-protocol version of these files, from
before continuous discharge was promoted to repo root -- now archived at
`experiments/07_pulse_protocol_archive/`.)

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

## Follow-up: does the fix hold up across SOC availability?

The original SOC-availability sweep (`run_soc_sweep.py`, now deleted) found
accuracy was *worst* at the high-SOC interval (0.7-1.0: RF 98.96%/XGB
94.79%) and near-perfect (100%/100%) at the two middle intervals — but that
was measured on the leaky voltage-bin features above, so it mostly reflects
how much of the one-sided/leaky bins each SOC range happened to populate,
not genuine physics. Re-running the same 4-interval sweep
(`soc_sweep.py`, this folder) with the coverage-filtered features gives a
very different, and much more informative, picture:

| SOC Interval | RF Accuracy | XGB Accuracy | Surviving Bin(s) |
|---|---|---|---|
| 0.7-1.0 | 96.88% | 94.79% | `dV_dQ_V_3.4_3.3` |
| 0.5-0.8 | 83.33% | 83.33% | `dV_dQ_V_3.4_3.3` |
| 0.3-0.6 | — | — | **none** (filter dropped every bin) |
| 0.1-0.4 | 71.28% | 73.40% | `dV_dQ_V_3.3_3.2` |

Two findings worth flagging:

1. **Accuracy degrades monotonically as SOC drops** (96.9% → 83.3% → n/a →
   71.3%/73.4%), the opposite shape from the original (leaky) sweep, and
   the intuitive one — less of the shared voltage range gets sampled at
   lower SOC, so there's less honest signal to classify on.
2. **The one surviving bin isn't the same bin across intervals.** At
   0.7-1.0 and 0.5-0.8 it's `dV_dQ_V_3.4_3.3`; at 0.3-0.6 the >20%
   mutual-coverage filter drops *every* bin (neither chemistry reliably
   reaches a shared one at that narrower range, so there's no feature left
   to train on at all — `ml_pipeline.py` was hardened during this run to
   report that plainly instead of crashing on a zero-column array, see
   below); at 0.1-0.4 a *different* bin (`dV_dQ_V_3.3_3.2`) becomes the
   sole survivor. The "one honest voltage bin" this project has been
   relying on is not a fixed, universal feature — which is exactly the
   arbitrary-missingness problem: a real Stena capture that happens to
   land in the 0.3-0.6 window would have no usable signal under this
   feature scheme at all.

**Robustness fix made along the way:** `ml_pipeline.py`'s `run_ml_pipeline`
now checks for zero remaining feature columns after
`feature_engineering.py`'s coverage filter and returns a clear
"no usable features" result instead of letting `SimpleImputer` fail with an
opaque `ValueError: at least one array or dtype is required` on a
zero-column array. This is a real, expected outcome at narrow enough SOC
ranges, not an error condition.
