"""
optimize_empa, attempt 01: EMPA-ONLY baseline, carried over from
experiments/32_fixed_voltage_cutoff_soc_sweep/evaluate_per_soc_point_models_ratio_to_median.py
(exp07 dropped -- per the professor's new directive, this campaign
focuses exclusively on EMPA's balanced accuracy; exp07/SNL are no
longer in scope). This is the starting point for this campaign, not new
work -- see RESULTS.md for the full history of how this combination was
found (negative-electrode-balance fix + ratio-to-median scale-invariant
features + per-chemistry coverage fix).

Per-SOC-point specialized Random Forest models, trained on a SCALE-
INVARIANT feature: each bin's dV/dQ divided by that SAME ROW's own
median dV/dQ across its own observed bins ("ratio-to-median"), instead
of absolute dV/dQ (which scales with physical cell size -- not usable
at the Stena production line, where a battery arrives at unknown
SOC/SOH with only a partial window observed and no total-capacity
baseline to normalize against).

SELF-CONTAINED: imports its own local copy of feature_engineering.py
(this folder) rather than the project-root one, so this pipeline logic
can be modified for this experiment without touching the global
pipeline. See this folder's RESULTS.md for the other local copies kept
here for full reproducibility (simulate_batteries_continuous_discharge_truncated.py,
simulate_lfp_negative_electrode_balance.py, combine_lfp_fix_with_baseline_nmc.py,
build_real_empa_soc_sweep_soh_range.py, ml_pipeline.py, soc_truncation.py)
-- none of those are invoked by this script; they're kept so the data
this script reads is itself reproducible from local files alone.

DATA SOURCES (both copied to this campaign's own data/optimize_empa_01_...
namespace, byte-for-byte, so this folder is runnable independently of
any other experiment's data path):
- Synthetic: data/optimize_empa_01_synthetic_baseline/ -- a byte-for-
  byte copy of data/32_synthetic_lfp_balance_fix_combined/ (negative-
  electrode-balance-fixed LFP, factor=0.7, + unchanged baseline NMC).
- Real: data/optimize_empa_01_real_empa/ -- a byte-for-byte copy of
  data/26_real_empa_soc_sweep_soh_range/ (the actual downloaded EMPA
  RO-Crate data; content unchanged, only the path is campaign-local).

METHOD: for each of the 12 Initial_SOC points, filter synthetic AND
real to that one SOC point, extract raw dV/dQ bins via
feature_engineering.py's standard >=20%-coverage filter, THEN re-check
coverage separately per (DataKind, Chemistry) group (the shared
utility's filter alone lets a bin survive even when one chemistry's
synthetic side has ZERO coverage there -- confirmed this corrupted
EMPA's Initial_SOC 0.5-0.7 folds in experiment 32; see RESULTS.md),
THEN divide every row by its own median across its own observed bins
(dropping rows with fewer than MIN_BINS_PER_ROW valid bins), THEN train
one Random Forest on that fold's synthetic rows and evaluate on that
fold's real rows.

Usage: python3 optimize_empa/experiment/01_ratio_to_median_baseline/run_evaluation.py
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
sys.path.insert(0, os.path.join(SCRIPT_DIR, "pipeline"))

from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "optimize_empa_01_synthetic_baseline", "raw", "advanced_synthetic_battery_data.csv")
REAL_EMPA_CSV = os.path.join(PROJECT_DIR, "data", "optimize_empa_01_real_empa", "raw", "real_empa_soc_sweep_raw.csv")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

MIN_BINS_PER_ROW = 3  # guard against zero/near-zero-variance rows producing NaN/inf ratios

FEATURES_DIR = os.path.join(SCRIPT_DIR, "outputs", "features")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "outputs", "per_soc_results.json")


def count_cells_and_cycles(real_csv):
    df = pd.read_csv(real_csv, usecols=["Variation_ID"], low_memory=False)
    vids = df["Variation_ID"].astype(str)
    cycle_ids = vids.str.split("_soc_").str[0]
    cell_ids = cycle_ids.str.split("_cycle_").str[0]
    return cell_ids.nunique(), cycle_ids.nunique()


def build_fold_csv(soc, synth_df, real_df):
    fold_csv = os.path.join(SCRIPT_DIR, "outputs", f"sim_and_real_raw_EMPA_soc_{soc}.csv")
    os.makedirs(os.path.dirname(fold_csv), exist_ok=True)
    synth_fold = synth_df[synth_df["Initial_SOC"] == soc].copy()
    real_fold = real_df[real_df["Initial_SOC"] == soc].copy()
    synth_fold["DataKind"] = "synthetic"
    real_fold["DataKind"] = "real"
    combined = pd.concat([synth_fold, real_fold], ignore_index=True)
    combined.to_csv(fold_csv, index=False)
    return fold_csv


def extract_fold_features(soc, fold_csv):
    features_df = create_features_by_voltage_bins(
        fold_csv, output_dir=os.path.join(FEATURES_DIR, f"soc_{soc}"), exclude_final_transition=True,
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


def filter_bins_per_chemistry_coverage(features_df, bin_cols, min_coverage=0.2):
    """The shared feature_engineering.py filter only checks POOLED
    (synthetic+real, both chemistries combined) coverage -- a bin can
    clear that threshold even when ONE (DataKind, Chemistry) group has
    ZERO coverage there, as long as the pooled average is carried by the
    other groups. See this campaign's RESULTS.md for the EMPA
    Initial_SOC 0.5-0.7 corruption this caused in experiment 32. Re-
    checks coverage separately for all four (DataKind, Chemistry) groups
    and drops any bin where the worst of the four falls under
    min_coverage."""
    keep = []
    dropped = []
    for col in bin_cols:
        group_coverage = features_df.groupby(['DataKind', 'Chemistry'])[col].apply(lambda s: s.notna().mean())
        if group_coverage.min() >= min_coverage:
            keep.append(col)
        else:
            dropped.append((col, group_coverage.to_dict()))
    return keep, dropped


def apply_ratio_to_median(features_df, bin_cols):
    """Row-wise (per sample, synthetic and real alike), using only that
    sample's own non-NaN observed bins. Rows with fewer than
    MIN_BINS_PER_ROW valid bins are dropped entirely."""
    X = features_df[bin_cols].values.astype(float)
    n_valid = np.sum(~np.isnan(X), axis=1)
    keep_mask = n_valid >= MIN_BINS_PER_ROW

    X_kept = X[keep_mask]
    row_median = np.nanmedian(X_kept, axis=1, keepdims=True)
    ratio = X_kept / row_median

    out = features_df.loc[keep_mask].copy()
    out[bin_cols] = ratio
    return out, int((~keep_mask).sum())


def run_one_fold(soc, synth_df, real_df):
    fold_csv = build_fold_csv(soc, synth_df, real_df)
    features_df, bin_cols = extract_fold_features(soc, fold_csv)

    if not bin_cols:
        print(f"  SOC={soc}: no surviving bins -- skipped.")
        return None

    bin_cols, dropped_bins = filter_bins_per_chemistry_coverage(features_df, bin_cols)
    for col, coverage in dropped_bins:
        print(f"  SOC={soc}: dropped bin {col} -- per-(DataKind,Chemistry) coverage {coverage}")
    if not bin_cols:
        print(f"  SOC={soc}: no bins survive the per-chemistry coverage check -- skipped.")
        return None

    features_df, n_dropped = apply_ratio_to_median(features_df, bin_cols)
    if n_dropped:
        print(f"  SOC={soc}: dropped {n_dropped} rows with < {MIN_BINS_PER_ROW} valid bins")

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
        "n_dropped_min_bins": n_dropped,
        "raw_accuracy": float((preds == y_test).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "LFP_recall": float(recalls[0]), "NMC_recall": float(recalls[1]),
    }
    print(f"  SOC={soc:<5} bins={len(bin_cols)} {bin_cols} n_synth={len(synth)} n_real={len(real)} "
          f"raw={result['raw_accuracy']*100:.2f}% balanced={result['balanced_accuracy']*100:.2f}% "
          f"LFP_recall={result['LFP_recall']:.3f} NMC_recall={result['NMC_recall']:.3f}")
    return result


def main():
    print(f"Loading synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    synth_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    print(f"  -> {len(synth_df)} rows")

    n_cells, n_cycles = count_cells_and_cycles(REAL_EMPA_CSV)
    print(f"EMPA: {n_cells} unique physical cells, {n_cycles} unique discharge cycles "
          f"(pooled across all Initial_SOC points)")
    real_df = pd.read_csv(REAL_EMPA_CSV, low_memory=False)

    buckets = {}
    for soc in SOC_START_POINTS:
        result = run_one_fold(soc, synth_df, real_df)
        if result is not None:
            buckets[str(soc)] = result

    all_results = {"EMPA": {"buckets": buckets, "n_cells": n_cells, "n_cycles": n_cycles}}
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}\nSaved results to {RESULTS_JSON}\n{'=' * 70}")

    print(f"\n=== SUMMARY (ratio-to-median, per-SOC-specialized Random Forest, EMPA balanced accuracy) ===")
    for soc in SOC_START_POINTS:
        bucket = buckets.get(str(soc))
        val = f"{bucket['balanced_accuracy']*100:.2f}%" if bucket else "--"
        print(f"  Initial_SOC={soc:<6} {val}")


if __name__ == "__main__":
    main()
