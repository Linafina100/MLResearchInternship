"""
Sim-to-real evaluation against the EMPA RO-Crate dataset (data/Dataset-
rocrate/, 199 real coin cells, 167 NMC / 32 LFP) -- a genuinely
independent real dataset from the one experiments 07/16/17/18/20/21 have
always tested against (and calibrated the NMC diffusivity/10 and LFP OCP
rate=-3 fixes to). Answers the question this experiment was created to
ask: do those two fixes generalize beyond the single real dataset they
were fit to, or were they overfit to its idiosyncrasies?

Trains on the SAME already-validated synthetic data experiment 21 built
(SOH=0.8, C-rate 0.1-0.2, NMC diffusivity/10, LFP OCP rate=-3 --
data/21_soh_0.8_lfp_ocp_tuned_v1.5/, no resimulation needed) and tests
against build_real_empa_dataset.py's output: real EMPA discharge cycles
selected at SOH=0.8+/-0.02 (see that script's docstring for why C-rate
is NOT also filtered to 0.1-0.2 here -- no cycle in this dataset
satisfies both simultaneously, and doing so would silently drop LFP
from the real test set entirely since every real LFP cell here was
cycled at a fixed ~1.0-1.07C).

Reports BALANCED accuracy and per-class recall alongside raw accuracy
(experiment 20 Part B's hard-won lesson: raw accuracy alone is
misleading whenever the real test set's classes are imbalanced -- this
one is 2407 LFP / 8586 NMC = 78.1% NMC, even more skewed than before).

Usage: python3 experiments/22_empa_rocrate_sim_to_real/evaluate_empa_rocrate_sim_to_real.py
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

sys.path.insert(0, PROJECT_DIR)  # for root ml_pipeline.py and feature_engineering.py

from ml_pipeline import run_ml_pipeline
from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "21_soh_0.8_lfp_ocp_tuned_v1.5", "raw", "advanced_synthetic_battery_data.csv")
REAL_RAW_CSV = os.path.join(PROJECT_DIR, "data", "22_real_empa_soh_0.8", "raw", "real_empa_raw.csv")
RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
BIN_COL_RE = re.compile(r"dV_dQ_V_([\d.]+)_([\d.]+)")


def build_combined_raw_csv():
    print(f"Loading experiment 21's LFP-OCP-tuned synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"  -> {len(sim_df)} rows")

    print(f"\nLoading real EMPA RO-Crate data from '{REAL_RAW_CSV}' "
          f"(build_real_empa_dataset.py output -- run that first if missing)...")
    real_df = pd.read_csv(REAL_RAW_CSV, low_memory=False)
    real_df["DataKind"] = "real"
    print(f"  -> {len(real_df)} rows, {real_df['Variation_ID'].nunique()} real discharge cycles")

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
    battery_to_kind = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)

    all_bin_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    zone_bin_cols = target_zone_bin_cols(all_bin_cols)
    print(f"\nAll surviving bins: {all_bin_cols}")
    print(f"Target-zone ({TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V) bins kept for training: {zone_bin_cols}")
    dropped_cols = [c for c in all_bin_cols if c not in zone_bin_cols]
    if dropped_cols:
        print(f"Dropped (outside target zone): {dropped_cols}")

    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))
    return features_df, zone_bin_cols


def report_magnitude_and_coverage(features_df, zone_bin_cols):
    print(f"\n{'=' * 70}\nMagnitude/coverage check (real vs. synthetic, target zone)\n{'=' * 70}")
    for chem in ("LFP", "NMC"):
        real_sub = features_df[(features_df.DataKind == "real") & (features_df.Chemistry == chem)]
        synth_sub = features_df[(features_df.DataKind == "synthetic") & (features_df.Chemistry == chem)]
        print(f"\n--- {chem} ---")
        print("REAL  mean:", real_sub[zone_bin_cols].mean().round(2).to_dict())
        print("SYNTH mean:", synth_sub[zone_bin_cols].mean().round(2).to_dict())
        print("REAL  coverage:", real_sub[zone_bin_cols].notna().mean().round(3).to_dict())
        print("SYNTH coverage:", synth_sub[zone_bin_cols].notna().mean().round(3).to_dict())


def balanced_metrics(synthetic_csv, real_csv, feature_cols):
    """Replicates ml_pipeline.py's exact preprocessing but reports BOTH
    models' balanced accuracy and per-class recall, not just raw accuracy
    -- mandatory here given this real test set is 78.1% NMC. See
    experiment 20 Part B's RESULTS.md for why this check is non-optional.
    """
    synth, real = pd.read_csv(synthetic_csv), pd.read_csv(real_csv)
    X_train = synth[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = real[feature_cols].replace([np.inf, -np.inf], np.nan)

    le = LabelEncoder()
    y_train = le.fit_transform(synth['Chemistry'])
    y_test = le.transform(real['Chemistry'])

    lower, upper = X_train.quantile(0.01), X_train.quantile(0.99)
    X_train_c = X_train.clip(lower=lower, upper=upper, axis=1)
    X_test_c = X_test.clip(lower=lower, upper=upper, axis=1)

    imputer = SimpleImputer(strategy='median')
    X_train_i, X_test_i = imputer.fit_transform(X_train_c), imputer.transform(X_test_c)
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train_i), scaler.transform(X_test_i)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1),
    }
    out = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        recalls = recall_score(y_test, preds, average=None)
        out[name] = {
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "LFP_recall": recalls[0], "NMC_recall": recalls[1],
        }
    return out


def main():
    features_df, zone_bin_cols = extract_features_with_kind()

    if not zone_bin_cols:
        print("\nNo surviving bins inside the target zone -- cannot train. Stopping here.")
        return

    report_magnitude_and_coverage(features_df, zone_bin_cols)

    metadata_cols = [
        'Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC',
        'Target_Capacity_Ah', 'Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set',
    ]
    keep_cols = [c for c in metadata_cols if c in features_df.columns] + zone_bin_cols

    synthetic_csv = os.path.join(FEATURES_DIR, "synthetic_features_target_zone.csv")
    real_csv = os.path.join(FEATURES_DIR, "real_features_target_zone.csv")
    features_df[features_df['DataKind'] == 'synthetic'][keep_cols].to_csv(synthetic_csv, index=False)
    features_df[features_df['DataKind'] == 'real'][keep_cols].to_csv(real_csv, index=False)

    n_lfp_real = (features_df[features_df.DataKind == 'real'].Chemistry == 'LFP').sum()
    n_nmc_real = (features_df[features_df.DataKind == 'real'].Chemistry == 'NMC').sum()

    print(f"\nTraining on synthetic data (SOH=0.8, NMC diffusivity/10 + LFP OCP rate=-3), "
          f"testing on the INDEPENDENT real EMPA RO-Crate dataset -- features restricted to the "
          f"{TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V target zone only...")
    results = run_ml_pipeline(synthetic_csv=synthetic_csv, real_csv=real_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    balanced = balanced_metrics(synthetic_csv, real_csv, zone_bin_cols)

    print("\n=== SUMMARY ===")
    print(f"Target-zone bins used: {zone_bin_cols}")
    print(f"Real test set (EMPA RO-Crate, independent of the dataset the fixes were calibrated to): "
          f"{n_lfp_real} LFP + {n_nmc_real} NMC batteries "
          f"({n_nmc_real / (n_lfp_real + n_nmc_real) * 100:.1f}% NMC) -- always check balanced accuracy.\n")
    for model_name, raw_acc in results['model_accuracies'].items():
        b = balanced[model_name]
        print(f"  {model_name:<15} raw={raw_acc*100:.2f}%  balanced={b['balanced_accuracy']*100:.2f}%  "
              f"LFP_recall={b['LFP_recall']:.3f}  NMC_recall={b['NMC_recall']:.3f}")
    print(f"\nTop features: {sorted(results['feature_importances'].items(), key=lambda kv: kv[1], reverse=True)[:5]}")
    print("\nFor context: experiment 21 (calibrated to the ORIGINAL real LFP/NMC dataset) scored "
          "RF raw=86.27%/balanced=90.55% (LFP recall 0.990, NMC recall 0.821), "
          "XGB raw=96.31%/balanced=97.21% (LFP recall 0.990, NMC recall 0.955).")


if __name__ == "__main__":
    main()
