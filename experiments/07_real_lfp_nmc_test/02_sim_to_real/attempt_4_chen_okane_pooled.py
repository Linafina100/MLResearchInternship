"""
Follow-up to evaluate_sim_to_real_per_nmc_parameter_set.py, which found
Chen2020 and OKane2022 individually transfer well (RF ~84%) while
Mohtat2020 reproduces the pooled failure (RF 24.84%) almost exactly.
This tests whether pooling just the two good parameter sets
(Chen2020+OKane2022, excluding Mohtat2020) preserves that ~84% result
while keeping some parameter-set diversity, rather than committing to a
single parameter set.

Also prints BOTH models' full classification reports (not just the
best-performing one, which is all root ml_pipeline.py surfaces) to start
diagnosing why XGBoost didn't improve in the per-parameter-set test even
when Random Forest did.

Usage: python3 experiments/07_real_lfp_nmc_test/02_sim_to_real/attempt_4_chen_okane_pooled.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP07_DIR = os.path.dirname(SCRIPT_DIR)  # experiments/07_real_lfp_nmc_test/, for the shared parsers
PROJECT_DIR = os.path.dirname(os.path.dirname(EXP07_DIR))

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXP07_DIR)

from feature_engineering import create_features_by_voltage_bins
from parse_real_lfp import parse_lfp_discharge_files
from parse_real_nmc import parse_nmc_files

WORK_DIR = os.path.join(SCRIPT_DIR, "chen_okane_pooled")
os.makedirs(WORK_DIR, exist_ok=True)
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

SOC_INTERVAL_DIRS = [
    "continuous_soc_0.7-1.0", "continuous_soc_0.5-0.8",
    "continuous_soc_0.3-0.6", "continuous_soc_0.1-0.4",
]
KEEP_PARAMETER_SETS = ["Chen2020", "OKane2022"]  # excludes Mohtat2020


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Matches root ml_pipeline.py's 1st/99th percentile clip step."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_bound_ = X.quantile(self.lower)
        self.upper_bound_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(lower=self.lower_bound_, upper=self.upper_bound_, axis=1).values


def make_pipeline(clf):
    return Pipeline([
        ("clip", PercentileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def main():
    print("Loading experiment 03's already-simulated data (4 SOC intervals, no re-simulation)...")
    dfs = []
    for interval_dir in SOC_INTERVAL_DIRS:
        path = os.path.join(PROJECT_DIR, "data", interval_dir, "raw", "advanced_synthetic_battery_data.csv")
        df = pd.read_csv(path)
        print(f"  -> {interval_dir}: {len(df)} rows")
        dfs.append(df)
    all_synthetic_df = pd.concat(dfs, ignore_index=True)

    keep = (all_synthetic_df['Chemistry'] == 'LFP') | (all_synthetic_df['Base_Parameter_Set'].isin(KEEP_PARAMETER_SETS))
    sim_df = all_synthetic_df[keep].copy()
    sim_df["DataKind"] = "synthetic"
    print(f"\nKeeping NMC parameter sets {KEEP_PARAMETER_SETS} (dropping Mohtat2020): "
          f"{(sim_df['Chemistry'] == 'LFP').sum()} LFP rows, {(sim_df['Chemistry'] == 'NMC').sum()} NMC rows")

    print("\nParsing real LFP data...")
    lfp_df = parse_lfp_discharge_files()
    print("\nParsing real NMC data...")
    nmc_df = parse_nmc_files()
    real_df = pd.concat([lfp_df, nmc_df], ignore_index=True)
    real_df["DataKind"] = "real"

    combined = pd.concat([sim_df, real_df], ignore_index=True)
    raw_csv = os.path.join(WORK_DIR, "raw_chen_okane.csv")
    combined.to_csv(raw_csv, index=False)

    print("\nExtracting features (root feature_engineering.py, exclude_final_transition=True)...")
    features_dir = os.path.join(WORK_DIR, "features")
    features_df = create_features_by_voltage_bins(raw_csv, output_dir=features_dir, exclude_final_transition=True)

    raw_reload = pd.read_csv(raw_csv)
    raw_reload['Battery_ID'] = raw_reload.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_reload.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)

    bin_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    print(f"\nSurviving bins: {bin_cols}")
    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))

    train_df = features_df[features_df['DataKind'] == 'synthetic']
    test_df = features_df[features_df['DataKind'] == 'real']

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['Chemistry'])
    y_test = le.transform(test_df['Chemistry'])
    X_train, X_test = train_df[bin_cols], test_df[bin_cols]

    print(f"\n{'=' * 70}\nTraining on Chen2020+OKane2022 pooled ({len(X_train)} samples), "
          f"testing on real ({len(X_test)} samples)\n{'=' * 70}")

    for name, clf in [
        ("Random Forest", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8,
                                   colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)),
    ]:
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = (preds == y_test).mean()
        print(f"\n--- {name}: accuracy = {acc:.4f} ---")
        print(classification_report(y_test, preds, target_names=le.classes_))
        print("Confusion matrix (rows=true, cols=predicted, order=", list(le.classes_), "):")
        print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()
