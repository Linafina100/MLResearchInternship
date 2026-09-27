"""
Experiment 28: instead of one Random Forest trained on ALL 12 Initial_SOC
truncation points pooled together (experiments 26/27's methodology), train
a SEPARATE Random Forest for each Initial_SOC point, on all 4 real
datasets (exp07, EMPA, CALCE, SNL).

WHY: experiment 26/27's pooled model relies on some higher-voltage bins
(e.g. `dV_dQ_V_3.2_3.1`) that only survive the mutual-coverage filter
because the pool includes plenty of full-charge (Initial_SOC=1.0)
batteries. A single real battery truncated to a LOW Initial_SOC often has
zero real coverage in those bins (confirmed directly for EMPA: real LFP
coverage of the 3.2-3.3V bin drops from 100% to 33% between Initial_SOC
0.6 and 0.5) -- so those features go entirely median-imputed for it, which
measurably biases the pooled model's predictions (observed: LFP recall
drifts toward 0 as Initial_SOC drops, a "predict NMC for everything"
pattern). Training one model PER SOC point instead means each model's own
training data is truncated to the SAME degree as the real batteries it
will be evaluated against, so `feature_engineering.py`'s existing
>=20%-mutual-coverage filter naturally decides, per SOC point, which bins
both chemistries actually still reach at that specific truncation level --
narrowing to fewer, deeper bins at low SOC on its own, without hand-picking
a voltage range (see experiment 24's finding that letting the filter
decide, rather than hand-picking, tends to work best) and without
switching to rested-initialization training (which would reproduce
experiment 26's own "Attempt 1" collapse, an unrelated confound -- see
that experiment's RESULTS.md).

Uses the SAME synthetic training data (data/26_soh_range_continuous_discharge_truncated_v1/)
and the SAME 4 real per-dataset CSVs experiments 26/27 already built --
no resimulation, no new real data. Only the grouping of rows into a
training set changes: filtered to `Initial_SOC == soc` instead of pooled
across all 12 points.

Per-fold combined CSVs are always overwritten (never skip-if-exists) --
the pooled evaluate scripts' skip-if-exists caching caused stale-result
bugs twice earlier this session; each fold here is small enough that
always rebuilding costs little.

Usage: python3 experiments/28_per_soc_point_models/evaluate_per_soc_point_models.py
       python3 experiments/28_per_soc_point_models/evaluate_per_soc_point_models.py SNL   # just one dataset
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
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "26_soh_range_continuous_discharge_truncated_v1", "raw", "advanced_synthetic_battery_data.csv")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

DATASETS = {
    "exp07": os.path.join(PROJECT_DIR, "data", "26_real_exp07_soc_sweep", "raw", "real_exp07_soc_sweep_raw.csv"),
    "EMPA": os.path.join(PROJECT_DIR, "data", "26_real_empa_soc_sweep_soh_range", "raw", "real_empa_soc_sweep_raw.csv"),
    "CALCE": os.path.join(PROJECT_DIR, "data", "26_real_calce_soc_sweep", "raw", "real_calce_soc_sweep_raw.csv"),
    "SNL": os.path.join(PROJECT_DIR, "data", "27_real_snl_soc_sweep", "raw", "real_snl_soc_sweep_raw.csv"),
}

FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "per_soc_results.json")


def build_fold_csv(dataset_name, soc, synth_df, real_df):
    fold_csv = os.path.join(SCRIPT_DIR, f"sim_and_real_raw_{dataset_name}_soc_{soc}.csv")
    synth_fold = synth_df[synth_df["Initial_SOC"] == soc].copy()
    real_fold = real_df[real_df["Initial_SOC"] == soc].copy()
    synth_fold["DataKind"] = "synthetic"
    real_fold["DataKind"] = "real"
    combined = pd.concat([synth_fold, real_fold], ignore_index=True)
    combined.to_csv(fold_csv, index=False)  # always overwritten -- see module docstring
    return fold_csv, len(synth_fold), len(real_fold)


def extract_fold_features(dataset_name, soc, fold_csv):
    features_df = create_features_by_voltage_bins(
        fold_csv, output_dir=os.path.join(FEATURES_DIR, dataset_name, f"soc_{soc}"), exclude_final_transition=True,
    )
    raw_df = pd.read_csv(fold_csv, low_memory=False)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    id_cols = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(id_cols)

    bin_cols = sorted(
        (c for c in features_df.columns if c.startswith('dV_dQ_V_')),
        key=lambda c: -float(c.split('_')[3]),
    )
    return features_df, bin_cols


def run_one_fold(dataset_name, soc, synth_df, real_df):
    fold_csv, n_synth_rows, n_real_rows = build_fold_csv(dataset_name, soc, synth_df, real_df)
    features_df, bin_cols = extract_fold_features(dataset_name, soc, fold_csv)

    if not bin_cols:
        print(f"  SOC={soc}: no surviving bins -- skipped.")
        return None

    synth = features_df[features_df.DataKind == 'synthetic']
    real = features_df[features_df.DataKind == 'real']
    if len(synth) == 0 or len(real) == 0:
        print(f"  SOC={soc}: no synthetic or no real batteries survived feature extraction -- skipped.")
        return None

    X_train = synth[bin_cols].replace([np.inf, -np.inf], np.nan)
    le = LabelEncoder()
    y_train = le.fit_transform(synth['Chemistry'])
    if len(set(y_train)) < 2:
        print(f"  SOC={soc}: synthetic side missing one chemistry -- skipped.")
        return None

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

    result = {
        "bin_cols": bin_cols,
        "n_synth_batteries": len(synth),
        "n": len(real),
        "raw_accuracy": float((preds == y_test).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "LFP_recall": float(recalls[0]), "NMC_recall": float(recalls[1]),
    }
    print(f"  SOC={soc:<5} bins={len(bin_cols)} {bin_cols} n_synth={len(synth)} n_real={len(real)} "
          f"raw={result['raw_accuracy']*100:.2f}% balanced={result['balanced_accuracy']*100:.2f}% "
          f"LFP_recall={result['LFP_recall']:.3f} NMC_recall={result['NMC_recall']:.3f}")
    return result


def run_one_dataset(dataset_name, real_csv, synth_df):
    print(f"\n{'=' * 70}\n{dataset_name}\n{'=' * 70}")
    real_df = pd.read_csv(real_csv, low_memory=False)
    dataset_results = {}
    for soc in SOC_START_POINTS:
        result = run_one_fold(dataset_name, soc, synth_df, real_df)
        if result is not None:
            dataset_results[str(soc)] = result
    return dataset_results


def main():
    only = sys.argv[1:] or list(DATASETS.keys())
    print(f"Loading shared synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    synth_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    print(f"  -> {len(synth_df)} rows")

    results_path = RESULTS_JSON
    all_results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    for dataset_name in only:
        all_results[dataset_name] = run_one_dataset(dataset_name, DATASETS[dataset_name], synth_df)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}\nSaved all results to {results_path}\n{'=' * 70}")

    done = [d for d in ["exp07", "EMPA", "CALCE", "SNL"] if d in all_results]
    print(f"\n=== SUMMARY (per-SOC-specialized Random Forest, balanced accuracy) -- datasets: {done} ===")
    header = f"{'Initial_SOC':<12}" + "".join(f"{d:<12}" for d in done)
    print(header)
    for soc_key in [str(s) for s in SOC_START_POINTS]:
        row = f"{soc_key:<12}"
        for d in done:
            if soc_key in all_results[d]:
                row += f"{all_results[d][soc_key]['balanced_accuracy']*100:<12.2f}"
            else:
                row += f"{'--':<12}"
        print(row)


if __name__ == "__main__":
    main()
