"""
Experiment 27: Initial_SOC sweep, evaluated against FOUR independent real
datasets (experiment 07's original set, EMPA, CALCE, and the new SNL
dataset), **Random Forest only** -- experiment 26 already established
that XGBoost never exceeds ~54% balanced accuracy against EMPA anywhere
in this SOC grid (see experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/RESULTS.md);
this experiment doesn't repeat that dead end.

exp07/EMPA/CALCE's Random Forest numbers are NOT re-run here -- they are
read directly from experiment 26's already-computed
experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/sweep_results.json (only the SNL
physics/training data is new; nothing about those three datasets' own
pipeline changed). Only SNL gets a fresh training+evaluation pass here.

Same efficient pattern as experiment 26: per dataset, combine synthetic
(data/26_soh_range_continuous_discharge_truncated_v1/, continuous-
discharge-truncated -- reused unchanged, see that experiment's RESULTS.md
for why no resimulation was needed for SNL) with that dataset's full
SOC-sweep real CSV (all 12 Initial_SOC buckets pooled), extract features
ONCE, train Random Forest ONCE, then evaluate per Initial_SOC bucket
separately.

Reports BALANCED accuracy and per-class recall for every (dataset, SOC)
combination (experiment 20 Part B's mandatory lesson).

Usage: python3 experiments/22_29_sim_to_real_validation/27_snl_soc_sweep/evaluate_soc_sweep_four_datasets.py
       python3 experiments/22_29_sim_to_real_validation/27_snl_soc_sweep/evaluate_soc_sweep_four_datasets.py SNL   # just SNL
"""
import json
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
EXP26_DIR = os.path.join(PROJECT_DIR, "experiments", "22_29_sim_to_real_validation", "26_soc_sweep_three_datasets")
sys.path.insert(0, PROJECT_DIR)

from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "26_soh_range_continuous_discharge_truncated_v1", "raw", "advanced_synthetic_battery_data.csv")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

EXP26_RESULTS_JSON = os.path.join(EXP26_DIR, "sweep_results.json")
REUSED_DATASETS = ["exp07", "EMPA", "CALCE"]  # read from exp26, not re-run

DATASETS = {
    "SNL": os.path.join(PROJECT_DIR, "data", "27_real_snl_soc_sweep", "raw", "real_snl_soc_sweep_raw.csv"),
}

FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "sweep_results.json")


def build_combined_raw_csv(dataset_name, real_csv):
    combined_csv = os.path.join(SCRIPT_DIR, f"sim_and_real_raw_{dataset_name}.csv")
    if os.path.exists(combined_csv):
        print(f"  Note: '{combined_csv}' already exists -- reusing it.")
        return combined_csv

    print(f"  Loading synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"    -> {len(sim_df)} rows")

    print(f"  Loading real {dataset_name} SOC-sweep data from '{real_csv}'...")
    real_df = pd.read_csv(real_csv, low_memory=False)
    real_df["DataKind"] = "real"
    print(f"    -> {len(real_df)} rows, {real_df['Variation_ID'].nunique()} real SOC-sweep variants")

    combined = pd.concat([sim_df, real_df], ignore_index=True)
    combined.to_csv(combined_csv, index=False)
    return combined_csv


def extract_features_with_kind(dataset_name, real_csv):
    combined_csv = build_combined_raw_csv(dataset_name, real_csv)

    print(f"  Extracting features (root feature_engineering.py, all bins, >=20% mutual-coverage filter)...")
    features_df = create_features_by_voltage_bins(
        combined_csv, output_dir=os.path.join(FEATURES_DIR, dataset_name), exclude_final_transition=True,
    )

    raw_df = pd.read_csv(combined_csv, low_memory=False)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    id_cols = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')[['DataKind', 'Initial_SOC']]
    features_df['DataKind'] = features_df['Battery_ID'].map(id_cols['DataKind'])
    features_df['Initial_SOC'] = features_df['Battery_ID'].map(id_cols['Initial_SOC'])
    if 'Source_File' in raw_df.columns:
        src_col = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['Source_File']
        features_df['Source_File'] = features_df['Battery_ID'].map(src_col)

    bin_cols = sorted(
        (c for c in features_df.columns if c.startswith('dV_dQ_V_')),
        key=lambda c: -float(c.split('_')[3]),
    )
    print(f"  Surviving bins ({len(bin_cols)}): {bin_cols}")
    print(f"  " + features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries').to_string().replace("\n", "\n  "))
    return features_df, bin_cols


def evaluate_bucket(model, imputer, scaler, le, lower, upper, real_bucket_df, feature_cols):
    X_test = real_bucket_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_test = le.transform(real_bucket_df['Chemistry'])

    X_test_c = X_test.clip(lower=lower, upper=upper, axis=1)
    X_test_i = imputer.transform(X_test_c)
    X_test_s = scaler.transform(X_test_i)

    preds = model.predict(X_test_s)
    recalls = recall_score(y_test, preds, average=None, labels=[0, 1], zero_division=0)
    raw_acc = (preds == y_test).mean()
    return {
        "n": len(real_bucket_df),
        "raw_accuracy": float(raw_acc),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "LFP_recall": float(recalls[0]), "NMC_recall": float(recalls[1]),
    }


def run_one_dataset(dataset_name, real_csv):
    print(f"\n{'=' * 70}\n{dataset_name}\n{'=' * 70}")
    features_df, bin_cols = extract_features_with_kind(dataset_name, real_csv)

    if not bin_cols:
        print(f"  No surviving bins for {dataset_name} -- skipping.")
        return {}

    synth_df = features_df[features_df['DataKind'] == 'synthetic']
    X_train = synth_df[bin_cols].replace([np.inf, -np.inf], np.nan)
    le = LabelEncoder()
    y_train = le.fit_transform(synth_df['Chemistry'])
    print(f"  Target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    lower, upper = X_train.quantile(0.01), X_train.quantile(0.99)
    X_train_c = X_train.clip(lower=lower, upper=upper, axis=1)
    imputer = SimpleImputer(strategy='median')
    X_train_i = imputer.fit_transform(X_train_c)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    dataset_results = {}
    for soc in SOC_START_POINTS:
        real_bucket = features_df[(features_df.DataKind == 'real') & (features_df.Initial_SOC == soc)]
        if len(real_bucket) == 0:
            print(f"\n  --- Initial_SOC={soc}: no surviving real batteries, skipped ---")
            continue
        n_lfp = (real_bucket.Chemistry == 'LFP').sum()
        n_nmc = (real_bucket.Chemistry == 'NMC').sum()
        print(f"\n  --- Initial_SOC={soc}: n={len(real_bucket)} ({n_lfp} LFP + {n_nmc} NMC) ---")
        r = evaluate_bucket(model, imputer, scaler, le, lower, upper, real_bucket, bin_cols)
        dataset_results[str(soc)] = {"Random Forest": r}  # string key throughout -- JSON round-tripping
        print(f"    Random Forest  raw={r['raw_accuracy']*100:.2f}%  balanced={r['balanced_accuracy']*100:.2f}%  "
              f"LFP_recall={r['LFP_recall']:.3f}  NMC_recall={r['NMC_recall']:.3f}")

    return {"bin_cols": bin_cols, "buckets": dataset_results}


def load_reused_results():
    """Pull exp07/EMPA/CALCE's already-computed Random Forest numbers
    from experiment 26's sweep_results.json -- dropping XGBoost from the
    bucket dicts since this experiment is RF-only."""
    with open(EXP26_RESULTS_JSON) as f:
        exp26_results = json.load(f)

    reused = {}
    for name in REUSED_DATASETS:
        if name not in exp26_results:
            continue
        entry = exp26_results[name]
        rf_only_buckets = {
            soc: {"Random Forest": bucket["Random Forest"]}
            for soc, bucket in entry.get("buckets", {}).items()
        }
        reused[name] = {"bin_cols": entry["bin_cols"], "buckets": rf_only_buckets}
    return reused


def main():
    only = sys.argv[1:] or list(DATASETS.keys())  # e.g. `python3 ... SNL` (the default anyway)
    results_path = RESULTS_JSON
    all_results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for name, entry in load_reused_results().items():
        all_results[name] = entry

    for dataset_name in only:
        all_results[dataset_name] = run_one_dataset(dataset_name, DATASETS[dataset_name])
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}\nSaved all results to {results_path}\n{'=' * 70}")

    done = [d for d in ["exp07", "EMPA", "CALCE", "SNL"] if d in all_results]
    print(f"\n=== SUMMARY (Random Forest balanced accuracy) -- datasets: {done} ===")
    header = f"{'Initial_SOC':<12}" + "".join(f"{d:<12}" for d in done)
    print(header)
    for soc_key in [str(s) for s in SOC_START_POINTS]:
        row = f"{soc_key:<12}"
        for d in done:
            buckets = all_results[d].get("buckets", {})
            if soc_key in buckets:
                row += f"{buckets[soc_key]['Random Forest']['balanced_accuracy']*100:<12.2f}"
            else:
                row += f"{'--':<12}"
        print(row)


if __name__ == "__main__":
    main()
