# Hyperparameter tuning for SOC 0.1-0.4 (post artifact-fix)

## What this experiment is

Experiment 14 fixed the termination-step artifact by excluding each
battery's final raw transition. At SOC 0.1-0.4, this leaves genuine
accuracy of 73.68% (RF) / 75.79% (XGB) on a single surviving feature bin
(`dV_dQ_V_3.2_3.1`) — down from the artifact-inflated 94.74%/94.74%. This
experiment asks whether that genuine number improves by tuning RF/XGBoost
hyperparameters, instead of the current hand-picked defaults (RF:
`n_estimators=150, max_depth=10`; XGB: `n_estimators=150,
learning_rate=0.08, max_depth=4, subsample=0.8, colsample_bytree=0.8`).

Scope: SOC 0.1-0.4 only (the other three intervals are either already
near-ceiling or out of scope here), hyperparameters only (preprocessing
kept identical to root `ml_pipeline.py`), evaluated via nested
cross-validation rather than repeatedly testing configs against the same
held-out split. Reuses experiment 03's raw data (no re-simulation) and
root `feature_engineering.create_features_by_voltage_bins(exclude_final_transition=True)`
to reproduce experiment 14's exact 1-feature dataset.

**Method**: preprocessing (inf→NaN, 1st/99th percentile clip, median
impute, standard scale) is wrapped in an `sklearn.pipeline.Pipeline` so
it's correctly refit on each cross-validation fold rather than fit once
outside the search. Hyperparameters are selected via `GridSearchCV`
(`cv=StratifiedGroupKFold(n_splits=5)`, grouped by `Battery_ID`, matching
root `ml_pipeline.py`'s own split method) on the training portion only;
the single winning configuration per model is then evaluated on the
held-out test set exactly once — the same split root `ml_pipeline.py`
would produce, so the "before" baseline reproduces experiment 14's
numbers exactly as a correctness check.

## Results

| Model | Baseline (test) | Tuned (inner-CV) | Tuned (test) |
|---|---|---|---|
| Random Forest | 73.68% | 77.45% | **80.00%** |
| XGBoost | 75.79% | 77.19% | **80.00%** |

Winning hyperparameters:
- **Random Forest**: `max_depth=5, min_samples_leaf=10, n_estimators=300`
  (default was `max_depth=10`, `min_samples_leaf` unset/1, `n_estimators=150`)
- **XGBoost**: `learning_rate=0.01, max_depth=3, n_estimators=50, subsample=1.0`
  (default was `learning_rate=0.08, max_depth=4, n_estimators=150, subsample=0.8`)

95 test samples: baseline RF gets 70 correct, tuned RF/XGB get 76 correct
— a swing of 6 predictions.

## Interpretation

**Tuning recovers a modest, real improvement**: +6.3pp for RF, +4.2pp for
XGB, both converging to the same 80.00% test accuracy. The held-out test
accuracy is slightly *higher* than the inner-CV estimate for both models
(80.00% vs. 77.45%/77.19%), not lower — a reassuring sign this isn't the
"looks good in cross-validation, fails to generalize" failure mode the
nested-CV setup was specifically designed to catch. That said, 95 test
samples is small: a swing of 6 predictions is a real, directionally
consistent effect, but not a large-sample-certain one — worth treating as
"tuning helps a little" rather than "tuning solves the interval."

**Both winning configurations are *more conservative* than the current
defaults**, not more complex: RF's best `min_samples_leaf=10` (vs.
effectively 1 by default) and shallower relative depth, XGB's best
`learning_rate=0.01` and only 50 estimators (vs. 0.08/150) both point the
same direction — heavier regularization. This is consistent with a
specific, sensible explanation: the current defaults were presumably
chosen with richer, multi-bin feature sets from other SOC intervals in
mind (where `0.5-0.8` and `0.3-0.6` still have 2 surviving bins each), and
are somewhat over-parameterized for a dataset this small (377 training
rows) with only one feature — more trees and deeper splits just have more
room to fit noise in the training folds without adding real signal, since
there's only one number to threshold on.

**Bottom line**: hyperparameter tuning alone cannot fully close the gap
left by the artifact fix (94.74% before the fix was never real; 80% is
the honest ceiling found here, not the artifact-level number). The
remaining headroom, if any, more likely lies in genuinely new information
— e.g. recovering usable signal from a wider voltage range, or additional
features — not further tuning of these two model families on this single
bin. Not attempted here, per this experiment's scope (hyperparameters
only, per the plan this was built against).
