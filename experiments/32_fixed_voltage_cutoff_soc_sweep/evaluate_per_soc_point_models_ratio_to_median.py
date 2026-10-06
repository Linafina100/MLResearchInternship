"""
Experiment 32: per-SOC-point specialized Random Forest models for exp07
and EMPA ONLY (SNL and CALCE excluded -- see below), using a NEW,
SCALE-INVARIANT feature: each bin's dV/dQ divided by that SAME ROW's own
median dV/dQ across its own observed bins ("ratio-to-median"), instead
of the absolute dV/dQ value every other experiment in this project uses.

WHY: absolute dV/dQ scales with a cell's physical size (Ah) -- confirmed
visually in plot_average_discharge_curves.py's dV/dQ panel, where
exp07 (large cells), EMPA (coin cells), and SNL (small cylindrical cells)
show very different magnitudes even within one chemistry. At the Stena
Recycling production line, a battery arrives at unknown SOC/SOH with
only a partial discharge window observed -- there is no total-capacity
baseline to normalize against. Ratio-to-median needs only the observed
window itself: for each sample, divide every bin's dV/dQ by the MEDIAN
of that sample's own observed bins. No cross-sample, no cross-dataset,
no total-capacity information used.

VALIDATED FIRST AS A SMOKE TEST (smoke_test_scale_invariant_features.py,
previous commit) before this full run, per project convention (same
lesson as the negative-electrode-diffusivity dead end: check before
committing compute). Findings that shaped this run's scope:
- The standard >=20% mutual-coverage filter (NOT the 0.0 override used
  in the smoke test's first pass) is required -- it was what stabilized
  EMPA NMC, which otherwise swings to -149/+25 from a handful of
  extremely sparse bins.
- A minimum-3-valid-bins-per-row guard is required to avoid NaN/inf from
  near-zero-variance rows.
- Ratio-to-median aligns exp07's and EMPA's cross-dataset scale well,
  and preserves their LFP-vs-NMC separation reasonably at every
  Initial_SOC checked (0.8/0.5/0.2).
- SNL does NOT benefit the same way: at low SOC (0.2) its ratio-to-
  median curves go flat (~1.0) for both chemistries, a genuine SHAPE
  mismatch (not a scale mismatch) a row-wise transform cannot fix --
  consistent with SNL's already-documented, separate generalization gap
  ([[exp27_snl_generalization_gap]]). SNL IS EXCLUDED FROM THIS RUN for
  that reason -- this run is scoped to prove the scale-invariance
  concept on the two datasets where it has a real chance of working, not
  to re-litigate SNL's separate, already-understood problem.

FOUND AND FIXED AFTER THE FIRST FULL RUN: EMPA's Initial_SOC 0.7/0.6/0.5
folds swung wildly (53%/61%/86% balanced accuracy) despite the other six
bins' real-data medians staying nearly identical fold-to-fold. Root
cause, confirmed directly: feature_engineering.py's >=20%-coverage filter
checks POOLED (synthetic+real, both chemistries combined) coverage, not
per-chemistry -- so a bin can survive even when ONE chemistry's synthetic
side has ZERO coverage there, as long as the other chemistry/real side
carries the pooled average over 20%. Exactly this happened: synthetic
LFP had 0% coverage in `dV_dQ_V_3.3_3.2` from SOC=0.7 downward (the
negative-electrode-balance fix shrank LFP's effective capacity enough
that this shallow bin falls outside every synthetic LFP battery's
truncated window), while synthetic NMC had 100% coverage there -- so
every LFP training row got the exact same imputed placeholder in that
column (imputed from NMC's distribution, since no real LFP value
existed to impute from), a constant, uninformative, genuinely misleading
feature for exactly one class, while real LFP test cycles had honest,
varying values there the model never learned to interpret. THE FIX:
`filter_bins_per_chemistry_coverage()` below re-checks coverage
separately for all four (DataKind, Chemistry) groups and drops any bin
where the WORST of the four falls under the threshold -- applied on top
of (not replacing) the existing shared-utility filter, contained to this
experiment's own script.

DATA SOURCES (same as evaluate_per_soc_point_models.py, no rebuild):
- Synthetic: data/32_synthetic_lfp_balance_fix_combined/ (the current
  pipeline's negative-electrode-balance-fixed LFP + unchanged baseline
  NMC) -- tests scale-invariance as an addition on top of the current
  best pipeline, not in isolation from it.
- exp07: data/26_real_exp07_soc_sweep/
- EMPA: data/26_real_empa_soc_sweep_soh_range/

METHOD: identical fold structure to evaluate_per_soc_point_models.py
(filter synthetic AND real to one Initial_SOC point, extract raw dV/dQ
bins via feature_engineering.py's standard >=20%-coverage filter) --
the ONLY difference is an added step after feature extraction: every
row (synthetic and real alike) is divided by its own median across its
own observed bins, and rows with fewer than MIN_BINS_PER_ROW valid bins
are dropped, before training/testing the Random Forest on the
transformed features.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/evaluate_per_soc_point_models_ratio_to_median.py
       python3 experiments/32_fixed_voltage_cutoff_soc_sweep/evaluate_per_soc_point_models_ratio_to_median.py EMPA   # just one dataset
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

# SNL excluded -- see module docstring for why.
DATASETS = {
    "exp07": os.path.join(PROJECT_DIR, "data", "26_real_exp07_soc_sweep", "raw", "real_exp07_soc_sweep_raw.csv"),
    "EMPA": os.path.join(PROJECT_DIR, "data", "26_real_empa_soc_sweep_soh_range", "raw", "real_empa_soc_sweep_raw.csv"),
}

MIN_BINS_PER_ROW = 3  # guard against zero/near-zero-variance rows producing NaN/inf ratios

FEATURES_DIR = os.path.join(SCRIPT_DIR, "features_ratio_to_median")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "per_soc_results_ratio_to_median.json")


def count_cells_and_cycles(real_csv):
    df = pd.read_csv(real_csv, usecols=["Variation_ID"], low_memory=False)
    vids = df["Variation_ID"].astype(str)
    cycle_ids = vids.str.split("_soc_").str[0]
    cell_ids = cycle_ids.str.split("_cycle_").str[0]
    return cell_ids.nunique(), cycle_ids.nunique()


def build_fold_csv(dataset_name, soc, synth_df, real_df):
    fold_csv = os.path.join(SCRIPT_DIR, f"sim_and_real_raw_ratiomed_{dataset_name}_soc_{soc}.csv")
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


def filter_bins_per_chemistry_coverage(features_df, bin_cols, min_coverage=0.2):
    """The shared feature_engineering.py filter only checks POOLED
    (synthetic+real, both chemistries combined) coverage -- a bin can
    clear that threshold even when ONE (DataKind, Chemistry) group has
    ZERO coverage there, as long as the pooled average is carried by the
    other groups. Confirmed this happened to synthetic LFP in EMPA's
    3.3-3.2V bin at Initial_SOC 0.5-0.7 (see module docstring) and
    corrupted that feature for the entire LFP training class. Re-checks
    coverage separately for all four (DataKind, Chemistry) groups and
    drops any bin where the worst of the four falls under min_coverage."""
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


def run_one_fold(dataset_name, soc, synth_df, real_df):
    fold_csv = build_fold_csv(dataset_name, soc, synth_df, real_df)
    features_df, bin_cols = extract_fold_features(dataset_name, soc, fold_csv)

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
    print(f"\n=== SUMMARY (ratio-to-median, per-SOC-specialized Random Forest, balanced accuracy) -- datasets: {done} ===")
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
