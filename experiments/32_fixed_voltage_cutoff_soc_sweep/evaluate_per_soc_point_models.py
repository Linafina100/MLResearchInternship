"""
Experiment 32 (rebuilt, now with the negative-electrode-balance fix):
per-SOC-point specialized Random Forest models for exp07, EMPA, and SNL
(CALCE dropped, not requested), with a NATURAL end-of-discharge cutoff --
no artificial fixed voltage floor -- and LFP's negative-electrode
capacity rebalanced to fix a real dV/dQ shape mismatch.

HOW THIS EXPERIMENT GOT HERE, IN THREE STAGES:
1. An earlier version forced a single universal voltage cutoff (2.5V,
   EMPA's own observed stopping point) onto every dataset's real data AND
   the synthetic generator, training ONE pooled model (across all 12
   Initial_SOC points) per dataset. That forced coverage into deep bins
   (like 2.6-2.5V) real LFP cells don't naturally reach. A side-by-side
   comparison against experiments 23/24 (97-99% balanced accuracy on
   EMPA) traced the real gap to something else: pooling all 12
   Initial_SOC points into ONE training set measurably blurs the model's
   sensitivity to bins a heavily-truncated low-SOC synthetic example has
   no data in -- confirmed directly by retraining on ONLY the
   Initial_SOC=1.0 synthetic slice, which exactly reproduced experiment
   24's 99.14% on identical real data. This is exactly what experiment 28
   already found and fixed (per-SOC-point specialized models, not one
   pooled model) -- so this script extends that fix to all three
   datasets, with the fixed-cutoff idea dropped.
2. Per-SOC specialization alone still left a sharp cliff for EMPA below
   Initial_SOC=0.5 (49-53% balanced accuracy at 0.4 and below). Traced to
   a genuine physics mismatch, not a methodology problem: real EMPA LFP's
   steep dV/dQ dive happens by ~2.65V, but the synthetic calibration's
   equivalent steepness only appears near 2.3-2.5V -- a region real EMPA
   cells never reach (they stop at ~2.5V). Confirmed directly by
   comparing full-range (no coverage filter) median dV/dQ per bin,
   synthetic vs. real EMPA LFP, Initial_SOC=1.0.
3. THE FIX: `simulate_lfp_negative_electrode_balance.py` scales LFP's
   negative-electrode capacity down (factor=0.7, independent of the
   positive electrode's own scaling) -- a lever distinct from both
   already-ruled-out LFP-side levers
   ([[exp29_lfp_ocp_tail_ceiling]], [[exp29_negative_electrode_diffusivity_fix]]).
   `diagnose_empa_negative_electrode_balance.py` found this shifts the
   dive to ~2.67V (full-bin-range checked, no runaway side effect, unlike
   the diffusivity dead end), and this script's own evaluation (below)
   confirms it survives the real test: EMPA's Initial_SOC=0.4/0.3
   buckets jump from ~50% to ~99% balanced accuracy. NMC is untouched by
   this fix -- `combine_lfp_fix_with_baseline_nmc.py` reuses the
   existing baseline NMC rows unchanged.

DATA SOURCES:
- Synthetic: data/32_synthetic_lfp_balance_fix_combined/ (LFP: negative-
  electrode-balance-fixed, 1.5V natural solver cutoff, SOH 0.8-1.0 range;
  NMC: identical to data/26_soh_range_continuous_discharge_truncated_v1/,
  untouched)
- exp07: data/26_real_exp07_soc_sweep/ (each cycle's own raw stopping
  point = 0% SOC, unchanged)
- EMPA: data/26_real_empa_soc_sweep_soh_range/ (unchanged)
- SNL: data/27_real_snl_soc_sweep/ (unchanged, WITH the lead-in-artifact
  fix from experiment 31 baked into the already-built CSV on disk)

METHOD (identical to experiment 28's evaluate_per_soc_point_models.py,
restricted to exp07/EMPA/SNL): for each dataset and each of the 12
Initial_SOC points, filter BOTH synthetic and real data to that one SOC
point, run feature_engineering.py's >=20%-mutual-coverage filter on just
that fold (so it naturally decides which bins both chemistries actually
reach at THAT specific truncation depth -- no hand-picked target zone),
train one Random Forest on that fold's synthetic rows, evaluate on that
fold's real rows. Also records per-dataset total unique physical cells
and total discharge cycles (pooled across all 12 Initial_SOC points) for
the sweep plot's legend.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/evaluate_per_soc_point_models.py
       python3 experiments/32_fixed_voltage_cutoff_soc_sweep/evaluate_per_soc_point_models.py SNL   # just one dataset
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

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "32_synthetic_lfp_balance_fix_combined", "raw", "advanced_synthetic_battery_data.csv")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

DATASETS = {
    "exp07": os.path.join(PROJECT_DIR, "data", "26_real_exp07_soc_sweep", "raw", "real_exp07_soc_sweep_raw.csv"),
    "EMPA": os.path.join(PROJECT_DIR, "data", "26_real_empa_soc_sweep_soh_range", "raw", "real_empa_soc_sweep_raw.csv"),
    "SNL": os.path.join(PROJECT_DIR, "data", "27_real_snl_soc_sweep", "raw", "real_snl_soc_sweep_raw.csv"),
}

FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "per_soc_results.json")


def count_cells_and_cycles(real_csv):
    """Every build script in this project names Variation_ID as
    '{cell_id}_cycle_{cycle_number}_soc_{s}' -- split on those markers to
    recover unique physical-cell and unique-cycle counts, pooled across
    all 12 Initial_SOC points."""
    df = pd.read_csv(real_csv, usecols=["Variation_ID"], low_memory=False)
    vids = df["Variation_ID"].astype(str)
    cycle_ids = vids.str.split("_soc_").str[0]
    cell_ids = cycle_ids.str.split("_cycle_").str[0]
    return cell_ids.nunique(), cycle_ids.nunique()


def build_fold_csv(dataset_name, soc, synth_df, real_df):
    # Always overwritten (never skip-if-exists) -- see experiment 28's
    # docstring for why that caching pattern caused stale-result bugs.
    fold_csv = os.path.join(SCRIPT_DIR, f"sim_and_real_raw_{dataset_name}_soc_{soc}.csv")
    synth_fold = synth_df[synth_df["Initial_SOC"] == soc].copy()
    real_fold = real_df[real_df["Initial_SOC"] == soc].copy()
    synth_fold["DataKind"] = "synthetic"
    real_fold["DataKind"] = "real"
    combined = pd.concat([synth_fold, real_fold], ignore_index=True)
    combined.to_csv(fold_csv, index=False)
    return fold_csv


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
    fold_csv = build_fold_csv(dataset_name, soc, synth_df, real_df)
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
    n_cells, n_cycles = count_cells_and_cycles(real_csv)
    print(f"  {n_cells} unique physical cells, {n_cycles} unique discharge cycles (pooled across all Initial_SOC points)")

    real_df = pd.read_csv(real_csv, low_memory=False)
    dataset_results = {}
    for soc in SOC_START_POINTS:
        result = run_one_fold(dataset_name, soc, synth_df, real_df)
        if result is not None:
            dataset_results[str(soc)] = result
    return {"buckets": dataset_results, "n_cells": n_cells, "n_cycles": n_cycles}


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

    done = [d for d in DATASETS if d in all_results]
    print(f"\n=== SUMMARY (per-SOC-specialized Random Forest, balanced accuracy) -- datasets: {done} ===")
    header = f"{'Initial_SOC':<12}" + "".join(f"{d:<12}" for d in done)
    print(header)
    for soc_key in [str(s) for s in SOC_START_POINTS]:
        row = f"{soc_key:<12}"
        for d in done:
            bucket = all_results[d].get("buckets", {}).get(soc_key)
            val = f"{bucket['balanced_accuracy']*100:.2f}" if bucket else "--"
            row += f"{val:<12}"
        print(row)


if __name__ == "__main__":
    main()
