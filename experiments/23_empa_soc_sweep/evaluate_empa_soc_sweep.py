"""
Real-to-sim Initial_SOC sweep evaluation against the EMPA RO-Crate
dataset. Trains ONE model exactly as experiment 22 did (synthetic
SOH=0.8, C-rate 0.1-0.2, NMC diffusivity/10 + LFP OCP rate=-3 --
data/21_soh_0.8_lfp_ocp_tuned_v1.5/, no resimulation) and tests it
separately against each Initial_SOC bucket of
build_real_empa_soc_sweep_dataset.py's output, to see whether the
sim-to-real result holds up as the real test window starts further from
a full charge.

The s=1.0 bucket is, by construction, the exact same real cycles
experiment 22 tested against (build_real_empa_soc_sweep_dataset.py
reuses its cycle selection unchanged) -- its numbers here are a
correctness check on this adapted pipeline: they should reproduce
experiment 22's published 97.89% RF / 92.71% XGBoost balanced accuracy.

Reports BALANCED accuracy and per-class recall for every bucket
(experiment 20 Part B's hard-won lesson: raw accuracy alone is
misleading on an imbalanced real test set -- every bucket here is
~78% NMC, same as experiment 22).

Usage: python3 experiments/23_empa_soc_sweep/evaluate_empa_soc_sweep.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.insert(0, PROJECT_DIR)  # for root feature_engineering.py

from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "21_soh_0.8_lfp_ocp_tuned_v1.5", "raw", "advanced_synthetic_battery_data.csv")
REAL_RAW_CSV = os.path.join(PROJECT_DIR, "data", "23_real_empa_soc_sweep", "raw", "real_empa_soc_sweep_raw.csv")
RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
BIN_COL_RE = re.compile(r"dV_dQ_V_([\d.]+)_([\d.]+)")

# The first six sit inside the synthetic training data's own empirical
# Initial_SOC range (~0.50-0.99); 0.4/0.3 are extrapolation -- see
# build_real_empa_soc_sweep_dataset.py's docstring.
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
EXTRAPOLATION_POINTS = {0.4, 0.3}


def build_combined_raw_csv():
    print(f"Loading experiment 21's LFP-OCP-tuned synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"  -> {len(sim_df)} rows")

    print(f"\nLoading real EMPA SOC-sweep data from '{REAL_RAW_CSV}' "
          f"(build_real_empa_soc_sweep_dataset.py output -- run that first if missing)...")
    real_df = pd.read_csv(REAL_RAW_CSV, low_memory=False)
    real_df["DataKind"] = "real"
    print(f"  -> {len(real_df)} rows, {real_df['Variation_ID'].nunique()} real SOC-sweep variants")

    combined = pd.concat([sim_df, real_df], ignore_index=True)
    combined.to_csv(RAW_CSV, index=False)
    print(f"\nCombined raw data saved to '{RAW_CSV}': "
          f"{(combined['DataKind'] == 'synthetic').sum()} synthetic rows, "
          f"{(combined['DataKind'] == 'real').sum()} real rows")
    return combined


def target_zone_bin_cols(all_bin_cols):
    kept = []
    for col in all_bin_cols:
        m = BIN_COL_RE.match(col)
        bin_high, bin_low = float(m.group(1)), float(m.group(2))
        if bin_low >= TARGET_ZONE_MIN and bin_high <= TARGET_ZONE_MAX:
            kept.append(col)
    return kept


def extract_features_with_kind():
    if not os.path.exists(RAW_CSV):
        build_combined_raw_csv()
    else:
        print(f"Note: '{RAW_CSV}' already exists -- reusing it. Delete it to re-build from source.")

    print("\nExtracting features (root feature_engineering.py, exclude_final_transition=True, "
          ">=20% mutual-coverage filter)...")
    features_df = create_features_by_voltage_bins(RAW_CSV, output_dir=FEATURES_DIR, exclude_final_transition=True)

    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    id_cols = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')[['DataKind', 'Initial_SOC']]
    features_df['DataKind'] = features_df['Battery_ID'].map(id_cols['DataKind'])
    features_df['Initial_SOC'] = features_df['Battery_ID'].map(id_cols['Initial_SOC'])

    all_bin_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    zone_bin_cols = target_zone_bin_cols(all_bin_cols)
    print(f"\nAll surviving bins: {all_bin_cols}")
    print(f"Target-zone ({TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V) bins kept for training: {zone_bin_cols}")
    dropped_cols = [c for c in all_bin_cols if c not in zone_bin_cols]
    if dropped_cols:
        print(f"Dropped (outside target zone): {dropped_cols}")

    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))
    return features_df, zone_bin_cols


def report_magnitude_and_coverage(features_df, zone_bin_cols, soc):
    real_sub_all = features_df[(features_df.DataKind == "real") & (features_df.Initial_SOC == soc)]
    synth_sub_all = features_df[features_df.DataKind == "synthetic"]
    for chem in ("LFP", "NMC"):
        real_sub = real_sub_all[real_sub_all.Chemistry == chem]
        synth_sub = synth_sub_all[synth_sub_all.Chemistry == chem]
        print(f"  {chem} REAL  mean: {real_sub[zone_bin_cols].mean().round(2).to_dict()}  "
              f"coverage: {real_sub[zone_bin_cols].notna().mean().round(2).to_dict()}")
    print(f"  (for reference) SYNTH mean/coverage pooled across all Initial_SOC: "
          f"{synth_sub_all[zone_bin_cols].mean().round(2).to_dict()}")


def fit_models(X_train_s, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1),
    }
    for model in models.values():
        model.fit(X_train_s, y_train)
    return models


def evaluate_bucket(models, imputer, scaler, le, lower, upper, real_bucket_df, feature_cols):
    """Replicates ml_pipeline.py's exact preprocessing (fit on the full
    synthetic training set, applied here) but reports BOTH models'
    balanced accuracy and per-class recall for one real Initial_SOC
    bucket at a time. See experiment 20 Part B's RESULTS.md for why
    balanced accuracy (not raw) is mandatory on this imbalanced test set.
    """
    X_test = real_bucket_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_test = le.transform(real_bucket_df['Chemistry'])

    X_test_c = X_test.clip(lower=lower, upper=upper, axis=1)
    X_test_i = imputer.transform(X_test_c)
    X_test_s = scaler.transform(X_test_i)

    out = {}
    for name, model in models.items():
        preds = model.predict(X_test_s)
        recalls = recall_score(y_test, preds, average=None, labels=[0, 1])
        out[name] = {
            "n": len(real_bucket_df),
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "LFP_recall": recalls[0], "NMC_recall": recalls[1],
        }
    return out


def main():
    features_df, zone_bin_cols = extract_features_with_kind()

    if not zone_bin_cols:
        print("\nNo surviving bins inside the target zone -- cannot train. Stopping here.")
        return

    synth_df = features_df[features_df['DataKind'] == 'synthetic']
    print(f"\n{'=' * 70}\nTraining once on the full synthetic set "
          f"({len(synth_df)} batteries, SOH=0.8, NMC diffusivity/10 + LFP OCP rate=-3), "
          f"features restricted to the {TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V target zone.\n{'=' * 70}")

    X_train = synth_df[zone_bin_cols].replace([np.inf, -np.inf], np.nan)
    le = LabelEncoder()
    y_train = le.fit_transform(synth_df['Chemistry'])
    print(f"Target classes mapped: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    lower, upper = X_train.quantile(0.01), X_train.quantile(0.99)
    X_train_c = X_train.clip(lower=lower, upper=upper, axis=1)
    imputer = SimpleImputer(strategy='median')
    X_train_i = imputer.fit_transform(X_train_c)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)

    models = fit_models(X_train_s, y_train)

    print(f"\n{'=' * 70}\nEvaluating per Initial_SOC bucket\n{'=' * 70}")
    bucket_results = {}
    for soc in SOC_START_POINTS:
        real_bucket = features_df[(features_df.DataKind == 'real') & (features_df.Initial_SOC == soc)]
        if len(real_bucket) == 0:
            print(f"\n--- Initial_SOC={soc}: no surviving real batteries, skipped ---")
            continue
        tag = " (EXTRAPOLATION: below synthetic training range ~0.50-0.99)" if soc in EXTRAPOLATION_POINTS else ""
        print(f"\n--- Initial_SOC={soc}{tag} ---")
        report_magnitude_and_coverage(features_df, zone_bin_cols, soc)
        result = evaluate_bucket(models, imputer, scaler, le, lower, upper, real_bucket, zone_bin_cols)
        bucket_results[soc] = result
        n_lfp = (real_bucket.Chemistry == 'LFP').sum()
        n_nmc = (real_bucket.Chemistry == 'NMC').sum()
        print(f"  n={len(real_bucket)} ({n_lfp} LFP + {n_nmc} NMC)")
        for model_name, r in result.items():
            print(f"  {model_name:<15} balanced={r['balanced_accuracy']*100:.2f}%  "
                  f"LFP_recall={r['LFP_recall']:.3f}  NMC_recall={r['NMC_recall']:.3f}")

    print(f"\n{'=' * 70}\n=== SWEEP SUMMARY ===\n{'=' * 70}")
    print(f"Target-zone bins used: {zone_bin_cols}\n")
    header = f"{'Initial_SOC':<12}{'n':<10}{'RF balanced':<14}{'RF LFP rec':<12}{'RF NMC rec':<12}{'XGB balanced':<14}{'XGB LFP rec':<12}{'XGB NMC rec':<12}"
    print(header)
    for soc in SOC_START_POINTS:
        if soc not in bucket_results:
            continue
        rf, xgb = bucket_results[soc]["Random Forest"], bucket_results[soc]["XGBoost"]
        tag = "*" if soc in EXTRAPOLATION_POINTS else " "
        print(f"{soc}{tag:<11}{rf['n']:<10}{rf['balanced_accuracy']*100:<14.2f}{rf['LFP_recall']:<12.3f}"
              f"{rf['NMC_recall']:<12.3f}{xgb['balanced_accuracy']*100:<14.2f}{xgb['LFP_recall']:<12.3f}{xgb['NMC_recall']:<12.3f}")
    print("\n* = extrapolation: below the synthetic training data's own Initial_SOC range (~0.50-0.99)")
    print("\nFor reference, experiment 22's published result on this same cycle selection "
          "(untruncated, i.e. this sweep's Initial_SOC=1.0 bucket) was "
          "RF balanced=97.89% (LFP recall 1.000, NMC recall 0.958), "
          "XGB balanced=92.71% (LFP recall 0.864, NMC recall 0.990).")


if __name__ == "__main__":
    main()
