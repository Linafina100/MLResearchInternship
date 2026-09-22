"""
Experiment 24: sim-to-real evaluation against the EMPA RO-Crate dataset,
same as experiment 22's evaluate_empa_rocrate_sim_to_real.py, but with
two deliberate differences:

1. Both synthetic and real data span SOH 0.8-1.0 (a RANGE, not experiment
   21/22's fixed 0.8 point) -- trains on
   simulate_batteries_soh_range.py's output, tests against
   build_real_empa_soh_range_dataset.py's output. Answers whether the
   validated NMC-diffusivity/LFP-OCP fixes generalize across a healthier
   slice of battery life, or were implicitly tuned to SOH=0.8 alone.

2. NO target-zone restriction. Experiments 22/23 both hardcoded training
   to just the 2.5-3.0V bins. Here we use whatever bins survive
   feature_engineering.py's >=20% mutual-coverage filter across the FULL
   1.9-4.3V span (its own defaults, unmodified) -- the filter itself
   stays on (disabling it would reopen the median-imputation leakage bug
   from experiments/01_leakage_fix_20_percent_coverage/RESULTS.md), we
   just don't additionally restrict to a hardcoded sub-range on top of
   it. A full 24-bin per-chemistry coverage table is printed before the
   filter runs so it's visible which bins pass/fail and why, and that
   set is compared directly against exp22's SOH=0.8 baseline (only the
   2.5-3.0V zone ever survived there, because LFP's synthetic voltage
   never reaches above ~3.5V at SOH=0.8 -- a real physics ceiling, not
   an artifact; see RESULTS.md for whether widening the SOH range
   changes this).

Reports BALANCED accuracy and per-class recall alongside raw accuracy
(experiment 20 Part B's hard-won lesson: raw accuracy alone is
misleading whenever the real test set's classes are imbalanced).

Usage: python3 experiments/24_soh_range_0.8_1.0_sim_to_real/evaluate_soh_range_sim_to_real.py
"""
import matplotlib
matplotlib.use("Agg")

import os
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

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "24_soh_range_0.8_1.0_v1.5", "raw", "advanced_synthetic_battery_data.csv")
REAL_RAW_CSV = os.path.join(PROJECT_DIR, "data", "24_real_empa_soh_0.8_1.0", "raw", "real_empa_raw.csv")
RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

# exp22's baseline: only these 5 bins ever survived the coverage filter
# at fixed SOH=0.8. Printed for direct comparison, not used as a filter.
EXP22_SURVIVING_BINS = [
    "dV_dQ_V_3.0_2.9", "dV_dQ_V_2.9_2.8", "dV_dQ_V_2.8_2.7",
    "dV_dQ_V_2.7_2.6", "dV_dQ_V_2.6_2.5",
]


def build_combined_raw_csv():
    print(f"Loading experiment 24's SOH-range synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"  -> {len(sim_df)} rows, SOH range {sim_df['SOH'].min():.3f}-{sim_df['SOH'].max():.3f}")

    print(f"\nLoading real EMPA RO-Crate data from '{REAL_RAW_CSV}' "
          f"(build_real_empa_soh_range_dataset.py output -- run that first if missing)...")
    real_df = pd.read_csv(REAL_RAW_CSV, low_memory=False)
    real_df["DataKind"] = "real"
    print(f"  -> {len(real_df)} rows, {real_df['Variation_ID'].nunique()} real discharge cycles, "
          f"SOH range {real_df['SOH'].min():.3f}-{real_df['SOH'].max():.3f}")

    combined = pd.concat([sim_df, real_df], ignore_index=True)
    combined.to_csv(RAW_CSV, index=False)
    print(f"\nCombined raw data saved to '{RAW_CSV}': "
          f"{(combined['DataKind'] == 'synthetic').sum()} synthetic rows, "
          f"{(combined['DataKind'] == 'real').sum()} real rows")
    return combined


def extract_features_with_kind():
    if not os.path.exists(RAW_CSV):
        build_combined_raw_csv()
    else:
        print(f"Note: '{RAW_CSV}' already exists -- reusing it. Delete it to re-build from source.")

    print("\nExtracting features (root feature_engineering.py, exclude_final_transition=True, "
          "default V_BIN_MIN=1.9, >=20% mutual-coverage filter, no additional target-zone restriction)...")
    features_df = create_features_by_voltage_bins(RAW_CSV, output_dir=FEATURES_DIR, exclude_final_transition=True)

    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)

    all_bin_cols = sorted(
        (c for c in features_df.columns if c.startswith('dV_dQ_V_')),
        key=lambda c: -float(c.split('_')[3]),
    )
    print(f"\nSurviving bins across the full 1.9-4.3V span ({len(all_bin_cols)}): {all_bin_cols}")
    new_bins = [c for c in all_bin_cols if c not in EXP22_SURVIVING_BINS]
    lost_bins = [c for c in EXP22_SURVIVING_BINS if c not in all_bin_cols]
    print(f"vs. experiment 22's SOH=0.8 baseline ({EXP22_SURVIVING_BINS}):")
    print(f"  newly surviving (didn't pass at SOH=0.8): {new_bins}")
    print(f"  no longer surviving: {lost_bins}")

    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))
    return features_df, all_bin_cols


def report_full_coverage_table(raw_csv):
    """Full per-chemistry coverage across all 24 candidate bins, before the
    mutual-coverage filter drops any -- the key diagnostic for whether
    widening SOH changed bin survival vs. exp22's SOH=0.8 baseline."""
    print(f"\n{'=' * 70}\nFull per-chemistry bin coverage (synthetic, unfiltered, min_chemistry_coverage=0)\n{'=' * 70}")
    unfiltered = create_features_by_voltage_bins(
        raw_csv, output_dir=os.path.join(FEATURES_DIR, "_coverage_check"), min_chemistry_coverage=0.0,
    )
    sim_unfiltered = unfiltered  # synthetic + real combined; coverage reported per DataKind below isn't
                                   # available here (DataKind mapping is done by the caller), so just show
                                   # per-chemistry coverage across everything in this raw CSV.
    bin_cols = sorted(
        (c for c in sim_unfiltered.columns if c.startswith('dV_dQ_V_')),
        key=lambda c: -float(c.split('_')[3]),
    )
    coverage = sim_unfiltered.groupby('Chemistry')[bin_cols].apply(lambda g: g.notna().mean())
    print(coverage.T.round(3).to_string())


def report_magnitude_and_coverage(features_df, bin_cols):
    print(f"\n{'=' * 70}\nMagnitude/coverage check (real vs. synthetic, all surviving bins)\n{'=' * 70}")
    for chem in ("LFP", "NMC"):
        real_sub = features_df[(features_df.DataKind == "real") & (features_df.Chemistry == chem)]
        synth_sub = features_df[(features_df.DataKind == "synthetic") & (features_df.Chemistry == chem)]
        print(f"\n--- {chem} ---")
        print("REAL  mean:", real_sub[bin_cols].mean().round(2).to_dict())
        print("SYNTH mean:", synth_sub[bin_cols].mean().round(2).to_dict())
        print("REAL  coverage:", real_sub[bin_cols].notna().mean().round(3).to_dict())
        print("SYNTH coverage:", synth_sub[bin_cols].notna().mean().round(3).to_dict())


def balanced_metrics(synthetic_csv, real_csv, feature_cols):
    """Replicates ml_pipeline.py's exact preprocessing but reports BOTH
    models' balanced accuracy and per-class recall, not just raw accuracy.
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
    report_full_coverage_table(SYNTHETIC_RAW_CSV if os.path.exists(SYNTHETIC_RAW_CSV) else RAW_CSV)

    features_df, bin_cols = extract_features_with_kind()

    if not bin_cols:
        print("\nNo surviving bins -- cannot train. Stopping here.")
        return

    report_magnitude_and_coverage(features_df, bin_cols)

    metadata_cols = [
        'Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC',
        'Target_Capacity_Ah', 'Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set',
    ]
    keep_cols = [c for c in metadata_cols if c in features_df.columns] + bin_cols

    synthetic_csv = os.path.join(FEATURES_DIR, "synthetic_features_all_bins.csv")
    real_csv = os.path.join(FEATURES_DIR, "real_features_all_bins.csv")
    features_df[features_df['DataKind'] == 'synthetic'][keep_cols].to_csv(synthetic_csv, index=False)
    features_df[features_df['DataKind'] == 'real'][keep_cols].to_csv(real_csv, index=False)

    n_lfp_real = (features_df[features_df.DataKind == 'real'].Chemistry == 'LFP').sum()
    n_nmc_real = (features_df[features_df.DataKind == 'real'].Chemistry == 'NMC').sum()

    print(f"\nTraining on SOH-range (0.8-1.0) synthetic data (NMC diffusivity/10 + LFP OCP rate=-3), "
          f"testing on the INDEPENDENT real EMPA RO-Crate dataset, SOH-range-matched -- "
          f"using all {len(bin_cols)} bins that survived the mutual-coverage filter...")
    results = run_ml_pipeline(synthetic_csv=synthetic_csv, real_csv=real_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    balanced = balanced_metrics(synthetic_csv, real_csv, bin_cols)

    print("\n=== SUMMARY ===")
    print(f"Bins used ({len(bin_cols)}): {bin_cols}")
    print(f"Real test set (EMPA RO-Crate, SOH in [0.8, 1.0]): "
          f"{n_lfp_real} LFP + {n_nmc_real} NMC batteries "
          f"({n_nmc_real / (n_lfp_real + n_nmc_real) * 100:.1f}% NMC) -- always check balanced accuracy.\n")
    for model_name, raw_acc in results['model_accuracies'].items():
        b = balanced[model_name]
        print(f"  {model_name:<15} raw={raw_acc*100:.2f}%  balanced={b['balanced_accuracy']*100:.2f}%  "
              f"LFP_recall={b['LFP_recall']:.3f}  NMC_recall={b['NMC_recall']:.3f}")
    print(f"\nTop features: {sorted(results['feature_importances'].items(), key=lambda kv: kv[1], reverse=True)[:5]}")
    print("\nFor context: experiment 22 (fixed SOH=0.8, target-zone-only bins) scored "
          "RF balanced=97.89%, XGB balanced=92.71% against this same EMPA RO-Crate dataset.")


if __name__ == "__main__":
    main()
