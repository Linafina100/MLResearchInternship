"""
Experiment 25: sim-to-real spot-check against the CALCE dataset -- the
SAME pattern experiments 22/24 used for EMPA (combine already-validated
synthetic data with a real raw CSV in the shape root feature_engineering.py
expects, run the shared feature/ml pipeline), but against a real dataset
that is fundamentally too small to support a generalization claim: only 4
physical cells (2 LFP, 2 NMC; build_calce_dataset.py's docstring covers the
column-collision bug fixed to recover their capacity data).

This is NOT presented as a third validated dataset the way EMPA is --
n=4 means a single flipped prediction swings the score by 25 points, and
one of the two NMC files (SP20-3) never discharges below 3.49V, so it has
no data at all in any bin the coverage filter keeps (see the per-battery
coverage printout below) and gets entirely median-imputed. This script
exists to answer one narrow question: does the CURRENT best pipeline
(experiment 24's SOH-range synthetic data, NMC diffusivity/10, LFP OCP
rate=-3) still get these 4 cells right, compared to the OLD exp06-based
pipeline's corrected 75% (LFP 2/2, NMC 1/2) found by a prior local,
unpushed bug-fix (`git show 3328d42`)?

Usage: python3 experiments/25_calce_sim_to_real/evaluate_calce_sim_to_real.py
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

sys.path.insert(0, PROJECT_DIR)

from ml_pipeline import run_ml_pipeline
from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "24_soh_range_0.8_1.0_v1.5", "raw", "advanced_synthetic_battery_data.csv")
REAL_RAW_CSV = os.path.join(PROJECT_DIR, "data", "25_calce_real", "raw", "calce_raw.csv")
RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']


def build_combined_raw_csv():
    print(f"Loading experiment 24's SOH-range synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"  -> {len(sim_df)} rows")

    print(f"\nLoading real CALCE data from '{REAL_RAW_CSV}' "
          f"(build_calce_dataset.py output -- run that first if missing)...")
    real_df = pd.read_csv(REAL_RAW_CSV, low_memory=False)
    real_df["DataKind"] = "real"
    print(f"  -> {len(real_df)} rows, {real_df['Variation_ID'].nunique()} real cells: "
          f"{real_df.drop_duplicates('Variation_ID')[['Variation_ID','Chemistry','Source_File']].to_string(index=False)}")

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
          "default V_BIN_MIN=1.9, >=20% mutual-coverage filter)...")
    features_df = create_features_by_voltage_bins(RAW_CSV, output_dir=FEATURES_DIR, exclude_final_transition=True)

    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)
    battery_to_srcfile = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['Source_File']
    features_df['Source_File'] = features_df['Battery_ID'].map(battery_to_srcfile)

    bin_cols = sorted(
        (c for c in features_df.columns if c.startswith('dV_dQ_V_')),
        key=lambda c: -float(c.split('_')[3]),
    )
    print(f"\nSurviving bins ({len(bin_cols)}): {bin_cols}")
    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))

    real_rows = features_df[features_df['DataKind'] == 'real']
    print(f"\nPer-real-cell coverage of surviving bins (NaN = median-imputed at train time):")
    print(real_rows[['Source_File', 'Chemistry'] + bin_cols].to_string(index=False))

    return features_df, bin_cols


def per_sample_predictions(synthetic_csv, real_csv, feature_cols, source_files):
    synth, real = pd.read_csv(synthetic_csv), pd.read_csv(real_csv)
    real = real.assign(Source_File=source_files)
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
    print(f"\n{'=' * 70}\nPer-sample predictions (n={len(real)} -- read individually, not as a percentage)\n{'=' * 70}")
    summary = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)
        pred_labels = le.inverse_transform(preds)
        true_labels = le.inverse_transform(y_test)
        print(f"\n--- {name} ---")
        n_correct = 0
        for i in range(len(real)):
            correct = pred_labels[i] == true_labels[i]
            n_correct += int(correct)
            conf = probs[i][preds[i]]
            print(f"  {real['Source_File'].iloc[i]:<45} actual={true_labels[i]:<3} "
                  f"predicted={pred_labels[i]:<3} confidence={conf:.3f}  {'CORRECT' if correct else 'WRONG'}")
        recalls = recall_score(y_test, preds, average=None, zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, preds)
        print(f"  -> {n_correct}/{len(real)} correct, raw accuracy={n_correct/len(real)*100:.2f}%, "
              f"balanced accuracy={bal_acc*100:.2f}%, LFP recall={recalls[0]:.3f}, NMC recall={recalls[1]:.3f}")
        summary[name] = {"n_correct": n_correct, "n_total": len(real), "balanced_accuracy": bal_acc,
                          "LFP_recall": recalls[0], "NMC_recall": recalls[1]}
    return summary


def main():
    features_df, bin_cols = extract_features_with_kind()

    if not bin_cols:
        print("\nNo surviving bins -- cannot train. Stopping here.")
        return

    # NOTE: 'Source_File' deliberately excluded from keep_cols -- ml_pipeline.py's
    # own metadata_cols list doesn't know about it, so it would otherwise be
    # treated as a (string) feature column and crash the numeric clip/impute
    # step. Passed to per_sample_predictions separately instead.
    metadata_cols = ['Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC',
                      'Target_Capacity_Ah', 'Ambient_Temperature_C', 'Resistance_Factor',
                      'Base_Parameter_Set']
    keep_cols = [c for c in metadata_cols if c in features_df.columns] + bin_cols

    real_source_files = features_df[features_df['DataKind'] == 'real']['Source_File'].tolist()

    synthetic_csv = os.path.join(FEATURES_DIR, "synthetic_features.csv")
    real_csv = os.path.join(FEATURES_DIR, "real_features.csv")
    features_df[features_df['DataKind'] == 'synthetic'][keep_cols].to_csv(synthetic_csv, index=False)
    features_df[features_df['DataKind'] == 'real'][keep_cols].to_csv(real_csv, index=False)

    print(f"\nTraining on experiment 24's SOH-range synthetic data, "
          f"testing on all {len(features_df[features_df.DataKind=='real'])} real CALCE cells...")
    results = run_ml_pipeline(synthetic_csv=synthetic_csv, real_csv=real_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    summary = per_sample_predictions(synthetic_csv, real_csv, bin_cols, real_source_files)

    print("\n=== SUMMARY ===")
    print(f"Bins used ({len(bin_cols)}): {bin_cols}")
    for model_name, raw_acc in results['model_accuracies'].items():
        s = summary[model_name]
        print(f"  {model_name:<15} {s['n_correct']}/{s['n_total']} correct  raw={raw_acc*100:.2f}%  "
              f"balanced={s['balanced_accuracy']*100:.2f}%  LFP_recall={s['LFP_recall']:.3f}  NMC_recall={s['NMC_recall']:.3f}")
    print("\nFor context: the prior LOCAL, UNPUSHED bug-fix against the OLD exp06-based synthetic "
          "data (git show 3328d42, branch verify-experiment-09-locally) found 75.00% (RF=XGB), "
          "LFP 2/2, NMC 1/2, on the same 4 real cells.")
    print("n=4: read this as a spot-check, not a generalization claim -- a single flipped "
          "prediction changes the score by 25 points.")


if __name__ == "__main__":
    main()
