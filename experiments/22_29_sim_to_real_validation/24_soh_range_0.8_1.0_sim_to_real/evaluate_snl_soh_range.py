"""
Curiosity check, not a new numbered experiment: does the SNL dataset do
any better under experiment 24's methodology (this project's
best-performing setup -- RF 99.14% against EMPA) than it did under the
continuous-discharge-truncated SOC-sweep methodology (experiments 27/28,
where it topped out around 42-64% balanced accuracy)?

Same rested-init synthetic training data experiment 24 already validated
against EMPA (data/24_soh_range_0.8_1.0_v1.5/, Initial_SOC 0.5-1.0,
SOH 0.8-1.0) -- unchanged, reused as-is. This pairing is safe (not
experiment 26's "Attempt 1" rested-vs-continuous trap): that collapse was
specifically caused by rested batteries at Initial_SOC<0.5; this dataset
never goes below 0.5.

Real SNL data: reuses experiments/22_29_sim_to_real_validation/27_snl_soc_sweep's already-built
data/27_real_snl_soc_sweep/raw/real_snl_soc_sweep_raw.csv, filtered to
Initial_SOC==1.0 -- per truncate_at_soc()'s own logic this means NO
truncation at all, i.e. exactly the untouched full discharge (already
SOH-filtered, already fixed for the doubled-capacity BOL artifact and the
fixed-time-sampling knee-resolution problem, already resampled onto an
80-point capacity grid) that a fresh exp24-style SNL build script would
produce. No new build script, no resimulation. Note: this slice went
through experiment 27's MAX_CYCLES_PER_CELL=20 subsampling, so it's not
literally every SOH-qualifying real cycle -- bounded like every other real
dataset in this project, not an exhaustive scan.

No target-zone restriction -- same as experiment 24's own approach, let
feature_engineering.py's >=20% mutual-coverage filter decide which bins
survive. Random Forest only (XGBoost dropped -- exp24's original script
predates the "XGBoost isn't fit for this task" finding from experiment 26,
established project-wide since).

Usage: python3 experiments/22_29_sim_to_real_validation/24_soh_range_0.8_1.0_sim_to_real/evaluate_snl_soh_range.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, PROJECT_DIR)

from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "24_soh_range_0.8_1.0_v1.5", "raw", "advanced_synthetic_battery_data.csv")
SNL_SOC_SWEEP_CSV = os.path.join(PROJECT_DIR, "data", "27_real_snl_soc_sweep", "raw", "real_snl_soc_sweep_raw.csv")
COMBINED_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw_snl.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features_snl")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']


def build_combined_raw_csv():
    print(f"Loading experiment 24's SOH-range synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"  -> {len(sim_df)} rows, SOH range {sim_df['SOH'].min():.3f}-{sim_df['SOH'].max():.3f}, "
          f"Initial_SOC range {sim_df['Initial_SOC'].min():.3f}-{sim_df['Initial_SOC'].max():.3f}")

    print(f"\nLoading real SNL SOC-sweep data from '{SNL_SOC_SWEEP_CSV}', filtering to Initial_SOC==1.0 "
          f"(untruncated full discharge, per truncate_at_soc()'s own logic)...")
    real_df = pd.read_csv(SNL_SOC_SWEEP_CSV, low_memory=False)
    real_df = real_df[real_df["Initial_SOC"] == 1.0].copy()
    real_df["DataKind"] = "real"
    print(f"  -> {len(real_df)} rows, {real_df['Variation_ID'].nunique()} real discharge cycles, "
          f"SOH range {real_df['SOH'].min():.3f}-{real_df['SOH'].max():.3f}")

    combined = pd.concat([sim_df, real_df], ignore_index=True)
    combined.to_csv(COMBINED_CSV, index=False)  # always overwritten, cheap to rebuild
    return combined


def extract_features_with_kind():
    build_combined_raw_csv()
    print("\nExtracting features (root feature_engineering.py, exclude_final_transition=True, "
          "no target-zone restriction -- same as experiment 24's own approach)...")
    features_df = create_features_by_voltage_bins(COMBINED_CSV, output_dir=FEATURES_DIR, exclude_final_transition=True)

    raw_df = pd.read_csv(COMBINED_CSV, low_memory=False)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)

    bin_cols = sorted(
        (c for c in features_df.columns if c.startswith('dV_dQ_V_')),
        key=lambda c: -float(c.split('_')[3]),
    )
    print(f"\nSurviving bins ({len(bin_cols)}): {bin_cols}")
    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))
    return features_df, bin_cols


def report_coverage_and_magnitude(features_df, bin_cols):
    print(f"\n{'=' * 70}\nPer-chemistry coverage and mean dV/dQ (real vs. synthetic)\n{'=' * 70}")
    for chem in ("LFP", "NMC"):
        real_sub = features_df[(features_df.DataKind == "real") & (features_df.Chemistry == chem)]
        synth_sub = features_df[(features_df.DataKind == "synthetic") & (features_df.Chemistry == chem)]
        print(f"\n--- {chem} ---")
        print("REAL  coverage:", real_sub[bin_cols].notna().mean().round(3).to_dict())
        print("SYNTH coverage:", synth_sub[bin_cols].notna().mean().round(3).to_dict())
        print("REAL  mean:", real_sub[bin_cols].mean().round(2).to_dict())
        print("SYNTH mean:", synth_sub[bin_cols].mean().round(2).to_dict())


def main():
    features_df, bin_cols = extract_features_with_kind()
    if not bin_cols:
        print("\nNo surviving bins -- cannot train. Stopping here.")
        return

    report_coverage_and_magnitude(features_df, bin_cols)

    synth = features_df[features_df.DataKind == 'synthetic']
    real = features_df[features_df.DataKind == 'real']
    n_lfp_real = (real.Chemistry == 'LFP').sum()
    n_nmc_real = (real.Chemistry == 'NMC').sum()

    X_train = synth[bin_cols].replace([np.inf, -np.inf], np.nan)
    le = LabelEncoder()
    y_train = le.fit_transform(synth['Chemistry'])

    lower, upper = X_train.quantile(0.01), X_train.quantile(0.99)
    X_train_c = X_train.clip(lower=lower, upper=upper, axis=1)
    imputer = SimpleImputer(strategy='median')
    X_train_i = imputer.fit_transform(X_train_c)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    X_test = real[bin_cols].replace([np.inf, -np.inf], np.nan)
    y_test = le.transform(real['Chemistry'])
    X_test_c = X_test.clip(lower=lower, upper=upper, axis=1)
    X_test_i = imputer.transform(X_test_c)
    X_test_s = scaler.transform(X_test_i)
    preds = model.predict(X_test_s)
    recalls = recall_score(y_test, preds, average=None, labels=[0, 1], zero_division=0)
    raw_acc = (preds == y_test).mean()
    balanced_acc = balanced_accuracy_score(y_test, preds)

    importances = sorted(zip(bin_cols, model.feature_importances_), key=lambda kv: -kv[1])

    print(f"\n{'=' * 70}\n=== SUMMARY ===\n{'=' * 70}")
    print(f"Bins used ({len(bin_cols)}): {bin_cols}")
    print(f"Real test set (SNL, Initial_SOC=1.0, SOH in [0.8, 1.0]): "
          f"{n_lfp_real} LFP + {n_nmc_real} NMC batteries "
          f"({n_nmc_real / (n_lfp_real + n_nmc_real) * 100:.1f}% NMC)")
    print(f"  Random Forest  raw={raw_acc*100:.2f}%  balanced={balanced_acc*100:.2f}%  "
          f"LFP_recall={recalls[0]:.3f}  NMC_recall={recalls[1]:.3f}")
    print(f"\nTop features: {importances[:5]}")
    print("\nFor context: experiment 24 scored RF balanced=99.14% against EMPA under this exact "
          "same methodology. Experiment 28's per-SOC-specialized model scored SNL 63.72% balanced "
          "at Initial_SOC=1.0 under the continuous-discharge-truncated SOC-sweep methodology.")


if __name__ == "__main__":
    main()
