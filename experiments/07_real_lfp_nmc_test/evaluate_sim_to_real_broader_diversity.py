"""
Improvement attempt #2 for experiment 07's sim-to-real test. Attempt #1
(evaluate_sim_to_real_artifact_fix.py) found the termination-artifact fix
removes the most extreme synthetic-NMC dV/dQ outliers but leaves a
systematic ~5-10x magnitude gap vs. real NMC across most bins -- not
explained by that artifact. This attempt asks whether that gap narrows
with broader synthetic diversity, specifically wider C-rate coverage:
experiment 03's synthetic data (used so far) is a fixed 0.6C for every
battery, while experiment 06's already-simulated data
(data/const_random_soc_*/) draws C-rate uniformly from 0.2-1.0C per
battery -- including rates well below 0.6C, which should produce smaller
IR-drop-driven dV/dQ magnitudes closer to real NMC's small values, if
C-rate variety (not just parameter-set mismatch) is part of the gap.

This combines experiment 03's + experiment 06's raw data (8 SOC-interval
datasets total) as the synthetic training set -- no re-simulation needed,
both already exist on disk. Real data and the rest of the pipeline
(feature extraction via root feature_engineering.py with
exclude_final_transition=True, since attempt #1 showed that's harmless
and slightly reduces outliers) are unchanged from attempt #1, so any
accuracy difference is attributable to the added C-rate diversity.

Note: this widens C-rate coverage only. Experiment 06 uses the same
SOH/resistance range as experiment 03 (aligned in an earlier pass), so
this does NOT test wider SOH/resistance diversity or additional NMC
parameter sets -- those would need new simulation, not attempted here.

Variation_ID is prefixed per source ("exp03_"/"exp06_") before combining:
both datasets use small integer Variation_ID counters from a similar
generation scheme, and while an accidental Battery_ID merge would also
require SOH/Initial_SOC to coincidentally match to many decimal places
(very unlikely), the prefix removes the risk entirely rather than relying
on that.

Usage: python3 experiments/07_real_lfp_nmc_test/evaluate_sim_to_real_broader_diversity.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.insert(0, PROJECT_DIR)   # for root ml_pipeline.py and feature_engineering.py
sys.path.insert(0, SCRIPT_DIR)    # for parse_real_lfp.py / parse_real_nmc.py

from ml_pipeline import run_ml_pipeline
from feature_engineering import create_features_by_voltage_bins
from parse_real_lfp import parse_lfp_discharge_files
from parse_real_nmc import parse_nmc_files

RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw_broader.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features_broader_diversity")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

EXP03_SOC_INTERVAL_DIRS = [
    "continuous_soc_0.7-1.0", "continuous_soc_0.5-0.8",
    "continuous_soc_0.3-0.6", "continuous_soc_0.1-0.4",
]
EXP03_RAW_FILENAME = "advanced_synthetic_battery_data.csv"

EXP06_SOC_INTERVAL_DIRS = [
    "const_random_soc_0.7-1.0", "const_random_soc_0.5-0.8",
    "const_random_soc_0.3-0.6", "const_random_soc_0.1-0.4",
]
EXP06_RAW_FILENAME = "continuous_synthetic_battery_data.csv"


def build_combined_raw_csv():
    print("Loading experiment 03's fixed-0.6C synthetic data (4 SOC intervals)...")
    sim_dfs = []
    for interval_dir in EXP03_SOC_INTERVAL_DIRS:
        path = os.path.join(PROJECT_DIR, "data", interval_dir, "raw", EXP03_RAW_FILENAME)
        df = pd.read_csv(path)
        df["Variation_ID"] = "exp03_" + df["Variation_ID"].astype(str)
        print(f"  -> {interval_dir}: {len(df)} rows")
        sim_dfs.append(df)

    print("\nLoading experiment 06's randomized-C-rate (0.2-1.0C) synthetic data (4 SOC intervals)...")
    for interval_dir in EXP06_SOC_INTERVAL_DIRS:
        path = os.path.join(PROJECT_DIR, "data", interval_dir, "raw", EXP06_RAW_FILENAME)
        df = pd.read_csv(path)
        df["Variation_ID"] = "exp06_" + df["Variation_ID"].astype(str)
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

    print("\nTraining on synthetic data (exp03+exp06, wider C-rate), testing on real combined data (sim-to-real)...")
    results = run_ml_pipeline(synthetic_csv=synthetic_csv, real_csv=real_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    print("\n=== SUMMARY ===")
    print(f"Surviving bins: {bin_cols}")
    print(f"Accuracies: {results['model_accuracies']}")
    print(f"Top features: {sorted(results['feature_importances'].items(), key=lambda kv: kv[1], reverse=True)[:5]}")


if __name__ == "__main__":
    main()
