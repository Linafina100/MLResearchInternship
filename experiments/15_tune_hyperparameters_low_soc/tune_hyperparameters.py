"""
Hyperparameter tuning for the SOC 0.1-0.4 classifier, on top of experiment
14's fix (excluding each battery's final raw transition -- see
experiments/14_exclude_final_transition/RESULTS.md). After that fix, this
interval's genuine accuracy is 73.68% (RF) / 75.79% (XGB) on a single
surviving feature bin (dV_dQ_V_3.2_3.1) -- down from the artifact-inflated
94.74%/94.74%. This script asks whether tuning RF/XGBoost hyperparameters
recovers any of that gap, using the current hand-picked defaults (RF:
n_estimators=150, max_depth=10; XGB: n_estimators=150, learning_rate=0.08,
max_depth=4, subsample=0.8, colsample_bytree=0.8) as the baseline.

Preprocessing (inf->NaN, 1st/99th percentile clipping, median imputation,
standard scaling) is kept identical to root ml_pipeline.py -- only the
classifier hyperparameters are searched. To avoid the leakage/multiple-
comparisons risk of repeatedly testing configs against the same held-out
split, hyperparameters are selected via cross-validation on the training
portion only (grouped by Battery_ID, matching root ml_pipeline.py's own
split method), and the single winning config per model is evaluated on
the held-out test set exactly once.

No re-simulation needed -- reuses experiment 03's already-committed raw
CSV for SOC 0.1-0.4 and root feature_engineering.py
(exclude_final_transition=True) to reproduce experiment 14's exact
1-feature dataset for this interval.

Usage: python3 experiments/15_tune_hyperparameters_low_soc/tune_hyperparameters.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)  # for root feature_engineering.py

from feature_engineering import create_features_by_voltage_bins

DATA_CSV = os.path.join(PROJECT_DIR, "data", "03_continuous_soc_0.1-0.4", "raw", "advanced_synthetic_battery_data.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")

METADATA_COLS = [
    'Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC',
    'Target_Capacity_Ah',
    'Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set',
]

# Matches root ml_pipeline.py's defaults exactly -- the "before" baseline.
BASELINE_RF = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
BASELINE_XGB = XGBClassifier(
    n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1,
)

# max_features / colsample_bytree left out -- both are moot with 1 feature.
RF_GRID = {
    "clf__n_estimators": [50, 100, 150, 300],
    "clf__max_depth": [2, 3, 5, 10, None],
    "clf__min_samples_leaf": [1, 3, 5, 10],
}
XGB_GRID = {
    "clf__n_estimators": [50, 100, 150, 300],
    "clf__max_depth": [1, 2, 3, 4],
    "clf__learning_rate": [0.01, 0.05, 0.08, 0.15],
    "clf__subsample": [0.6, 0.8, 1.0],
}


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clips each column to its [lower, upper] quantile range, fit on
    whatever data it's given -- matches root ml_pipeline.py's clipping
    step exactly, but as a proper sklearn transformer so it refits
    per-fold inside a Pipeline/GridSearchCV instead of being fit once
    outside the search."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_bound_ = X.quantile(self.lower)
        self.upper_bound_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(lower=self.lower_bound_, upper=self.upper_bound_, axis=1).values


def load_data():
    print(f"Loading raw data from '{DATA_CSV}' (reused from experiment 03, no re-simulation)...")
    full_df = create_features_by_voltage_bins(DATA_CSV, output_dir=FEATURES_DIR, exclude_final_transition=True)
    feature_cols = [c for c in full_df.columns if c not in METADATA_COLS]
    print(f"Feature columns: {feature_cols}")

    X = full_df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y = full_df['Chemistry'].copy()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    groups = full_df['Battery_ID'].values

    # Identical split to root ml_pipeline.py -- reproduces experiment 14's
    # exact held-out test set for this interval.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y_encoded, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    groups_train = groups[train_idx]
    print(f"Split: {len(X_train)} train samples | {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test, groups_train, le.classes_


def make_pipeline(clf):
    return Pipeline([
        ("clip", PercentileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def evaluate_baseline(name, clf, X_train, y_train, X_test, y_test):
    pipe = make_pipeline(clf)
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"[baseline/{name}] test accuracy = {acc:.4f}")
    return acc


def tune_and_evaluate(name, clf, grid, X_train, y_train, groups_train, X_test, y_test):
    pipe = make_pipeline(clf)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
    search.fit(X_train, y_train, groups=groups_train)

    test_acc = accuracy_score(y_test, search.predict(X_test))
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"[tuned/{name}] best inner-CV accuracy = {search.best_score_:.4f}, "
          f"held-out test accuracy = {test_acc:.4f}")
    print(f"[tuned/{name}] best params: {best_params}")
    return search.best_score_, test_acc, best_params


def main():
    X_train, X_test, y_train, y_test, groups_train, classes = load_data()
    print(f"Classes: {list(classes)}\n")

    print("=" * 60)
    print("BASELINE (root ml_pipeline.py's current defaults)")
    print("=" * 60)
    baseline_rf_acc = evaluate_baseline("Random Forest", BASELINE_RF, X_train, y_train, X_test, y_test)
    baseline_xgb_acc = evaluate_baseline("XGBoost", BASELINE_XGB, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH (nested CV on the training portion)")
    print("=" * 60)
    rf_cv_acc, rf_test_acc, rf_best_params = tune_and_evaluate(
        "Random Forest", RandomForestClassifier(random_state=42, n_jobs=-1), RF_GRID,
        X_train, y_train, groups_train, X_test, y_test,
    )
    xgb_cv_acc, xgb_test_acc, xgb_best_params = tune_and_evaluate(
        "XGBoost", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1), XGB_GRID,
        X_train, y_train, groups_train, X_test, y_test,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15}{'Baseline test':>15}{'Tuned inner-CV':>17}{'Tuned test':>13}")
    print(f"{'Random Forest':<15}{baseline_rf_acc:>15.4f}{rf_cv_acc:>17.4f}{rf_test_acc:>13.4f}")
    print(f"{'XGBoost':<15}{baseline_xgb_acc:>15.4f}{xgb_cv_acc:>17.4f}{xgb_test_acc:>13.4f}")
    print(f"\nRF best params: {rf_best_params}")
    print(f"XGB best params: {xgb_best_params}")


if __name__ == "__main__":
    main()
