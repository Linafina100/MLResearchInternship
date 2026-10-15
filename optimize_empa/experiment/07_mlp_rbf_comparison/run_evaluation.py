"""
optimize_empa, experiment 07 (MLP + RBF-SVM comparison): same pipeline
as experiment 06 (one pooled model trained once on the full-SOC-range,
continuous-discharge-truncated synthetic set, evaluated at every real
EMPA Initial_SOC bucket) -- but trains two additional model types
alongside Random Forest and XGBoost: a Multi-Layer Perceptron and an
RBF-kernel SVM, so all four can be compared on one plot.

NOTE on "RBF Network": scikit-learn has no built-in RBF-network
classifier. Three options were considered (a classic KMeans-centers +
Gaussian-basis + linear-readout network; an RBFSampler kernel-
approximation pipeline; a plain RBF-kernel SVM) -- sklearn.svm.SVC(kernel="rbf")
was chosen. Labeled honestly throughout as "RBF SVM": it's a max-margin
kernel classifier, not a network architecture (no hidden/readout
layers), despite using the same radial basis function.

No data_generation/ here -- reuses experiment 06's already-generated
synthetic data and attempt 01's real EMPA data unchanged (same
precedent as experiment 05 reusing experiment 03's data). Nothing
about the data changes, only which models get trained on it.

Also persists all four fitted models + shared preprocessing (imputer,
scaler, label encoder, clip bounds, bin_cols) to outputs/models/.
Set SAVE_MODELS=0 to skip.

Usage: python3 optimize_empa/experiment/07_mlp_rbf_comparison/run_evaluation.py
"""
import matplotlib
matplotlib.use("Agg")

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

sys.path.insert(0, os.path.join(SCRIPT_DIR, "pipeline"))
from feature_engineering import create_features_by_voltage_bins

# Reuses experiment 06's synthetic data and attempt 01's real data --
# no regeneration, only the model set under test changes.
SYNTHETIC_RAW_CSV = os.environ.get("SYNTHETIC_RAW_CSV", os.path.join(PROJECT_DIR, "data", "optimize_empa_low_soc_06_synthetic", "raw", "advanced_synthetic_battery_data.csv"))
REAL_RAW_CSV = os.environ.get("REAL_RAW_CSV", os.path.join(PROJECT_DIR, "data", "optimize_empa_low_soc_01_real", "raw", "real_empa_raw.csv"))
RAW_CSV = os.path.join(SCRIPT_DIR, "outputs", "sim_and_real_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "outputs", "features")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "outputs", "sweep_results.json")
MODELS_DIR = os.path.join(SCRIPT_DIR, "outputs", "models")
SAVE_MODELS = os.environ.get("SAVE_MODELS", "1") not in ("0", "false", "False")
GROUPBY_COLS = ["Chemistry", "Size_Multiplier", "SOH", "Initial_SOC", "Variation_ID"]

MODEL_NAMES = ["Random Forest", "XGBoost", "MLP", "RBF SVM"]
MODEL_FILENAMES = {
    "Random Forest": "random_forest.joblib", "XGBoost": "xgboost.joblib",
    "MLP": "mlp.joblib", "RBF SVM": "rbf_svm.joblib",
}

# Standard 12-point grid used throughout this project.
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]


def build_combined_raw_csv():
    os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
    print(f"Loading synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"  -> {len(sim_df)} rows")

    print(f"\nLoading real EMPA data from '{REAL_RAW_CSV}'...")
    real_df = pd.read_csv(REAL_RAW_CSV, low_memory=False)
    real_df["DataKind"] = "real"
    print(f"  -> {len(real_df)} rows, {real_df['Variation_ID'].nunique()} real discharge cycles")

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

    print("\nExtracting features (exclude_final_transition=True, >=20% mutual-coverage filter, "
          "no target-zone restriction)...")
    features_df = create_features_by_voltage_bins(RAW_CSV, output_dir=FEATURES_DIR, exclude_final_transition=True)

    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df["Battery_ID"] = raw_df.groupby(GROUPBY_COLS).ngroup()
    id_cols = raw_df.drop_duplicates("Battery_ID").set_index("Battery_ID")[["DataKind", "Initial_SOC", "Variation_ID"]]
    features_df["DataKind"] = features_df["Battery_ID"].map(id_cols["DataKind"])
    features_df["Initial_SOC"] = features_df["Battery_ID"].map(id_cols["Initial_SOC"])
    features_df["Variation_ID"] = features_df["Battery_ID"].map(id_cols["Variation_ID"])

    all_bin_cols = sorted(
        (c for c in features_df.columns if c.startswith("dV_dQ_V_")),
        key=lambda c: -float(c.split("_")[3]),
    )
    print(f"\nSurviving bins across the full voltage span ({len(all_bin_cols)}): {all_bin_cols}")
    print(features_df.groupby(["DataKind", "Chemistry"]).size().rename("n_batteries"))
    return features_df, all_bin_cols


def fit_models(X_train_s, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                              max_iter=1000, early_stopping=True, random_state=42),
        "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    }
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(X_train_s, y_train)
    return models


def evaluate_bucket(models, imputer, scaler, le, lower, upper, real_bucket_df, feature_cols):
    X_test = real_bucket_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_test = le.transform(real_bucket_df["Chemistry"])

    X_test_c = X_test.clip(lower=lower, upper=upper, axis=1)
    X_test_i = imputer.transform(X_test_c)
    X_test_s = scaler.transform(X_test_i)

    out = {}
    for name, model in models.items():
        preds = model.predict(X_test_s)
        recalls = recall_score(y_test, preds, average=None, labels=[0, 1], zero_division=0)
        out[name] = {
            "n": int(len(real_bucket_df)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "LFP_recall": float(recalls[0]),
            "NMC_recall": float(recalls[1]),
        }
    return out


def save_models(models, imputer, scaler, le, lower, upper, bin_cols):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, MODEL_FILENAMES[name]))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump({"bin_cols": list(bin_cols)}, f, indent=2)
    with open(os.path.join(MODELS_DIR, "clip_bounds.json"), "w") as f:
        json.dump({"lower": lower.to_dict(), "upper": upper.to_dict()}, f, indent=2)
    print(f"Saved all {len(models)} models + shared preprocessing -> '{MODELS_DIR}'")


def main():
    features_df, bin_cols = extract_features_with_kind()

    if not bin_cols:
        print("\nNo surviving bins -- cannot train. Stopping here.")
        return

    synth_df = features_df[features_df["DataKind"] == "synthetic"]
    print(f"\n{'=' * 70}\nTraining once on the full pooled synthetic set "
          f"({len(synth_df)} batteries, Initial_SOC {synth_df['Initial_SOC'].min():.2f}-"
          f"{synth_df['Initial_SOC'].max():.2f}) using all {len(bin_cols)} surviving bins.\n{'=' * 70}")

    X_train = synth_df[bin_cols].replace([np.inf, -np.inf], np.nan)
    le = LabelEncoder()
    y_train = le.fit_transform(synth_df["Chemistry"])
    print(f"Target classes mapped: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    lower, upper = X_train.quantile(0.01), X_train.quantile(0.99)
    X_train_c = X_train.clip(lower=lower, upper=upper, axis=1)
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train_c)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)

    models = fit_models(X_train_s, y_train)

    if SAVE_MODELS:
        save_models(models, imputer, scaler, le, lower, upper, bin_cols)

    print(f"\n{'=' * 70}\nEvaluating per Initial_SOC bucket\n{'=' * 70}")
    bucket_results = {}
    for soc in SOC_START_POINTS:
        real_bucket = features_df[(features_df.DataKind == "real") & (features_df.Initial_SOC == soc)]
        if len(real_bucket) == 0:
            print(f"\n--- Initial_SOC={soc}: no surviving real batteries, skipped ---")
            continue

        result = evaluate_bucket(models, imputer, scaler, le, lower, upper, real_bucket, bin_cols)
        bucket_results[str(soc)] = result
        n_lfp = (real_bucket.Chemistry == "LFP").sum()
        n_nmc = (real_bucket.Chemistry == "NMC").sum()
        print(f"\n--- Initial_SOC={soc} --- n={len(real_bucket)} ({n_lfp} LFP + {n_nmc} NMC)")
        for model_name in MODEL_NAMES:
            r = result[model_name]
            print(f"  {model_name:<15} balanced={r['balanced_accuracy']*100:.2f}%  "
                  f"LFP_recall={r['LFP_recall']:.3f}  NMC_recall={r['NMC_recall']:.3f}")

    print(f"\n{'=' * 70}\n=== SWEEP SUMMARY (balanced accuracy %) ===\n{'=' * 70}")
    print(f"Bins used ({len(bin_cols)}): {bin_cols}\n")
    header = f"{'Initial_SOC':<12}{'n':<10}" + "".join(f"{m:<16}" for m in MODEL_NAMES)
    print(header)
    for soc in SOC_START_POINTS:
        s_str = str(soc)
        if s_str not in bucket_results:
            continue
        row = bucket_results[s_str]
        n = row[MODEL_NAMES[0]]["n"]
        line = f"{soc:<12}{n:<10}" + "".join(f"{row[m]['balanced_accuracy']*100:<16.2f}" for m in MODEL_NAMES)
        print(line)

    existing_results = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r") as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Note: Could not parse existing '{RESULTS_JSON}' ({e}). Rebuilding.")

    # Variation_ID encodes "<cell>_cycle_<N>_soc_<X>" -- strip the SOC
    # suffix first to count unique physical discharge CYCLES (not
    # cycle x SOC-point combos), then strip the cycle number too for
    # unique physical CELLS.
    real_all = features_df[features_df["DataKind"] == "real"]
    cycle_ids = real_all["Variation_ID"].astype(str).str.split("_soc_").str[0]
    n_cycles = int(cycle_ids.nunique())
    n_cells = int(cycle_ids.str.split("_cycle_").str[0].nunique())
    print(f"\nReal EMPA data used: {n_cells} physical cells, {n_cycles} discharge cycles.")

    existing_results["EMPA"] = {"buckets": bucket_results, "meta": {"n_cells": n_cells, "n_cycles": n_cycles}}
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(existing_results, f, indent=2)
    print(f"\nSaved EMPA sweep results to '{RESULTS_JSON}'")


if __name__ == "__main__":
    main()
