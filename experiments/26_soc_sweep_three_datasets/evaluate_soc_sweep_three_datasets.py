"""
Experiment 26: Initial_SOC sweep, evaluated against THREE independent
real datasets (experiment 07's original set, EMPA, CALCE), using ALL
voltage bins the mutual-coverage filter lets survive (root
feature_engineering.py, unmodified >=20% filter) -- not the hand-picked
2.5-3.0V target zone experiments 22/23 restricted to.

Same efficient pattern as experiments/23_empa_soc_sweep/evaluate_empa_soc_sweep.py:
per dataset, combine synthetic (data/24_soh_range_0.8_1.0_v1.5/, SOH
sampled 0.8-1.0) with that dataset's full SOC-sweep real CSV (all 8
Initial_SOC buckets pooled), extract features ONCE (bin survival is
computed pooling all buckets together -- this is why a bucket's number
can differ slightly from an equivalent single-SOH-point run elsewhere in
this project, a "pooled-data coverage-filter effect," not a bug, per
experiment 23's own finding), train RF+XGB ONCE on the synthetic rows,
then evaluate per Initial_SOC bucket separately. 3 datasets x 1 training
pass each = 3 full pipeline passes, not 24.

CALCE (n=4 physical cells) additionally gets per-sample predictions
printed for every SOC bucket -- an aggregate percentage on 4 points hides
more than it reveals (same reasoning as experiment 25).

Reports BALANCED accuracy and per-class recall for every (dataset, SOC)
combination (experiment 20 Part B's mandatory lesson) -- raw accuracy is
also recorded for the RESULTS.md table but is never the headline number.

CAVEATS baked into this script's output, not just RESULTS.md prose:
- experiment 07: only 4 physical cells (1 LFP + 3 NMC), a documented
  device-confound dataset (see
  experiments/07_real_lfp_nmc_test/01_real_vs_real_device_confound/RESULTS.md)
  -- mitigated but not eliminated here since the model never trains on
  this dataset (see build_real_exp07_soc_sweep.py's docstring), and its
  SOH is NOT filtered (unlike EMPA/CALCE) since it was never SOH-filtered
  in any prior experiment either.
- CALCE: n=4 physical cells, a spot-check, not generalization evidence.
- EMPA: 199 physical cells, the only one of the three with enough
  independent cells to support a real generalization claim about SOC
  robustness specifically.

Usage: python3 experiments/26_soc_sweep_three_datasets/evaluate_soc_sweep_three_datasets.py
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
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "24_soh_range_0.8_1.0_v1.5", "raw", "advanced_synthetic_battery_data.csv")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
EXTRAPOLATION_POINTS = {0.4, 0.3}

DATASETS = {
    "exp07": os.path.join(PROJECT_DIR, "data", "26_real_exp07_soc_sweep", "raw", "real_exp07_soc_sweep_raw.csv"),
    "EMPA": os.path.join(PROJECT_DIR, "data", "26_real_empa_soc_sweep_soh_range", "raw", "real_empa_soc_sweep_raw.csv"),
    "CALCE": os.path.join(PROJECT_DIR, "data", "26_real_calce_soc_sweep", "raw", "real_calce_soc_sweep_raw.csv"),
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
    X_test = real_bucket_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_test = le.transform(real_bucket_df['Chemistry'])

    X_test_c = X_test.clip(lower=lower, upper=upper, axis=1)
    X_test_i = imputer.transform(X_test_c)
    X_test_s = scaler.transform(X_test_i)

    out = {}
    for name, model in models.items():
        preds = model.predict(X_test_s)
        recalls = recall_score(y_test, preds, average=None, labels=[0, 1], zero_division=0)
        raw_acc = (preds == y_test).mean()
        out[name] = {
            "n": len(real_bucket_df),
            "raw_accuracy": float(raw_acc),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "LFP_recall": float(recalls[0]), "NMC_recall": float(recalls[1]),
        }
        if len(real_bucket_df) <= 10:  # CALCE: print every individual prediction
            pred_labels = le.inverse_transform(preds)
            true_labels = le.inverse_transform(y_test)
            for i in range(len(real_bucket_df)):
                src = real_bucket_df['Source_File'].iloc[i] if 'Source_File' in real_bucket_df.columns else real_bucket_df['Variation_ID'].iloc[i]
                correct = "CORRECT" if pred_labels[i] == true_labels[i] else "WRONG"
                print(f"      [{name}] {src}: actual={true_labels[i]} predicted={pred_labels[i]} {correct}")
    return out


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

    models = fit_models(X_train_s, y_train)

    dataset_results = {}
    for soc in SOC_START_POINTS:
        real_bucket = features_df[(features_df.DataKind == 'real') & (features_df.Initial_SOC == soc)]
        if len(real_bucket) == 0:
            print(f"\n  --- Initial_SOC={soc}: no surviving real batteries, skipped ---")
            continue
        tag = " (EXTRAPOLATION)" if soc in EXTRAPOLATION_POINTS else ""
        n_lfp = (real_bucket.Chemistry == 'LFP').sum()
        n_nmc = (real_bucket.Chemistry == 'NMC').sum()
        print(f"\n  --- Initial_SOC={soc}{tag}: n={len(real_bucket)} ({n_lfp} LFP + {n_nmc} NMC) ---")
        result = evaluate_bucket(models, imputer, scaler, le, lower, upper, real_bucket, bin_cols)
        dataset_results[str(soc)] = result  # string key throughout -- consistent with JSON round-tripping
        for model_name, r in result.items():
            print(f"    {model_name:<15} raw={r['raw_accuracy']*100:.2f}%  balanced={r['balanced_accuracy']*100:.2f}%  "
                  f"LFP_recall={r['LFP_recall']:.3f}  NMC_recall={r['NMC_recall']:.3f}")

    return {"bin_cols": bin_cols, "buckets": dataset_results}


def main():
    only = sys.argv[1:] or list(DATASETS.keys())  # e.g. `python3 ... CALCE exp07` to run a subset
    results_path = RESULTS_JSON
    all_results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    for dataset_name in only:
        all_results[dataset_name] = run_one_dataset(dataset_name, DATASETS[dataset_name])
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}\nSaved all results to {results_path}\n{'=' * 70}")

    done = list(all_results.keys())
    print(f"\n=== SUMMARY (balanced accuracy) -- datasets completed so far: {done} ===")
    header = f"{'Initial_SOC':<12}" + "".join(f"{d+' RF':<12}{d+' XGB':<12}" for d in done)
    print(header)
    for soc_key in [str(s) for s in SOC_START_POINTS]:
        row = f"{soc_key:<12}"
        for d in done:
            buckets = all_results[d].get("buckets", {})
            if soc_key in buckets:
                row += f"{buckets[soc_key]['Random Forest']['balanced_accuracy']*100:<12.2f}{buckets[soc_key]['XGBoost']['balanced_accuracy']*100:<12.2f}"
            else:
                row += f"{'--':<12}{'--':<12}"
        print(row)


if __name__ == "__main__":
    main()
