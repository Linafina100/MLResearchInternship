"""
Improvement attempt #1 for experiment 07's sim-to-real test (baseline:
24.84% accuracy, evaluate_real_data.py -- both RF and XGB collapse to
predicting "LFP" almost always, LFP recall 1.00 / NMC recall 0.01).

Diagnosis (see RESULTS.md): comparing real vs. synthetic mean dV/dQ per
shared voltage bin shows LFP is well matched (both small, roughly -0 to
-9 per bin), but synthetic NMC's magnitudes are 10-100x larger than real
NMC's (real: -0.3 to -8.4; synthetic: -2 to -352) -- consistent with
experiment 14's already-proven solver-termination artifact (PyBaMM's
event-triggered discharge cutoff produces one oversized final step,
exploding the dV/dQ estimate for whichever bin it lands in). Experiment
14 fixed this with an `exclude_final_transition` flag on **root**
`feature_engineering.py` -- but this experiment's synthetic-side feature
extraction uses `feature_engineering_continuous.py`
(experiments/03_continuous_discharge_soc_sweep/), a separate local copy
that never received that fix.

This script is evaluate_real_data.py with exactly one change: synthetic-
side feature extraction uses root feature_engineering.py's
`create_features_by_voltage_bins(..., exclude_final_transition=True)`
instead of feature_engineering_continuous.py's
`create_features_by_voltage_bins_continuous` (which has no such option).
Everything else -- real-data parsing, the combined-raw-then-extract-once
pattern, the sim-to-real evaluation -- is unchanged, so any accuracy
difference is attributable to this one fix.

Usage: python3 experiments/07_real_lfp_nmc_test/02_sim_to_real/attempt_1_artifact_fix.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP07_DIR = os.path.dirname(SCRIPT_DIR)  # experiments/07_real_lfp_nmc_test/, for the shared parsers
PROJECT_DIR = os.path.dirname(os.path.dirname(EXP07_DIR))

sys.path.insert(0, PROJECT_DIR)   # for root ml_pipeline.py and feature_engineering.py
sys.path.insert(0, EXP07_DIR)     # for parse_real_lfp.py / parse_real_nmc.py

from ml_pipeline import run_ml_pipeline
from feature_engineering import create_features_by_voltage_bins
from parse_real_lfp import parse_lfp_discharge_files
from parse_real_nmc import parse_nmc_files

RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")  # reused as-is from evaluate_real_data.py
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features_artifact_fix")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

SOC_INTERVAL_DIRS = [
    "continuous_soc_0.7-1.0", "continuous_soc_0.5-0.8",
    "continuous_soc_0.3-0.6", "continuous_soc_0.1-0.4",
]


def build_combined_raw_csv():
    print("Loading experiment 03's already-simulated data (4 SOC intervals, no re-simulation)...")
    sim_dfs = []
    for interval_dir in SOC_INTERVAL_DIRS:
        path = os.path.join(PROJECT_DIR, "data", interval_dir, "raw", "advanced_synthetic_battery_data.csv")
        df = pd.read_csv(path)
        print(f"  -> {interval_dir}: {len(df)} rows")
        sim_dfs.append(df)
    sim_df = pd.concat(sim_dfs, ignore_index=True)
    sim_df["DataKind"] = "synthetic"

    print("\nParsing real LFP data...")
    lfp_df = parse_lfp_discharge_files()
    print("\nParsing real NMC data...")
    nmc_df = parse_nmc_files()
    real_df = pd.concat([lfp_df, nmc_df], ignore_index=True)
    real_df["DataKind"] = "real"

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
          ">=20% mutual-coverage filter)...")
    features_df = create_features_by_voltage_bins(RAW_CSV, output_dir=FEATURES_DIR, exclude_final_transition=True)

    raw_df = pd.read_csv(RAW_CSV)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)

    bin_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    print(f"\nSurviving bins: {bin_cols}")
    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))
    return features_df, bin_cols


def print_magnitude_comparison(features_df, bin_cols):
    print(f"\n{'=' * 70}\nReal vs. synthetic mean dV/dQ per bin (magnitude-mismatch check)\n{'=' * 70}")
    for chem in ("LFP", "NMC"):
        real_mean = features_df[(features_df['DataKind'] == 'real') & (features_df['Chemistry'] == chem)][bin_cols].mean()
        sim_mean = features_df[(features_df['DataKind'] == 'synthetic') & (features_df['Chemistry'] == chem)][bin_cols].mean()
        print(f"\n--- {chem} ---")
        print("REAL  mean:", real_mean.round(2).to_dict())
        print("SYNTH mean:", sim_mean.round(2).to_dict())


def main():
    features_df, bin_cols = extract_features_with_kind()
    print_magnitude_comparison(features_df, bin_cols)

    synthetic_csv = os.path.join(FEATURES_DIR, "synthetic_features.csv")
    real_csv = os.path.join(FEATURES_DIR, "real_features.csv")
    features_df[features_df['DataKind'] == 'synthetic'].drop(columns=['DataKind']).to_csv(synthetic_csv, index=False)
    features_df[features_df['DataKind'] == 'real'].drop(columns=['DataKind']).to_csv(real_csv, index=False)

    print("\nTraining on synthetic data (artifact fix applied), testing on real combined data (sim-to-real)...")
    results = run_ml_pipeline(synthetic_csv=synthetic_csv, real_csv=real_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    print("\n=== SUMMARY ===")
    print(f"Surviving bins: {bin_cols}")
    print(f"Accuracies: {results['model_accuracies']}")
    print(f"Top features: {sorted(results['feature_importances'].items(), key=lambda kv: kv[1], reverse=True)[:5]}")


if __name__ == "__main__":
    main()
