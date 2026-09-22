"""
Real LFP vs. NMC classification test -- sim-to-real: train on simulated
data, test on real, physically-measured cells.

This replaces this script's original design, which combined the real LFP
(parse_real_lfp.py) and real NMC (parse_real_nmc.py) data and ran a plain
train/test split *within* that real data -- explicitly not a sim-to-real
transfer test, by its own prior docstring. That wasn't the intended
design for this experiment; this version is.

Real data: experiment 07's own parsers (parse_real_lfp.py,
parse_real_nmc.py), unchanged -- 579 LFP cycles + 1782 NMC cycles.

Simulated data: experiment 03's four already-simulated SOC-interval raw
datasets (data/continuous_soc_{0.7-1.0,0.5-0.8,0.3-0.6,0.1-0.4}/raw/),
concatenated for full discharge-range coverage -- a single narrow SOC
interval wouldn't span the same voltage range the real cells' full
discharge cycles do. No re-simulation needed.

Features are extracted ONCE from the combined (synthetic + real) raw
data via feature_engineering_continuous.py (unchanged), so the >=20%
mutual-coverage filter's bin-survival decision and the resulting feature
columns are consistent between the synthetic training set and the real
test set -- extracting them separately could produce different surviving
bins and break ml_pipeline.py's `X_test = df_real[feature_cols]` step. A
`DataKind` column ("synthetic"/"real") is reattached afterward by
replicating feature_engineering_continuous.py's own deterministic
Battery_ID assignment on the raw data (that function doesn't carry
arbitrary metadata columns through its per-battery aggregation).

Usage: python3 experiments/07_real_lfp_nmc_test/02_sim_to_real/evaluate_sim_to_real.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP07_DIR = os.path.dirname(SCRIPT_DIR)  # experiments/07_real_lfp_nmc_test/, for the shared parsers
PROJECT_DIR = os.path.dirname(os.path.dirname(EXP07_DIR))
CONTINUOUS_EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments", "03_continuous_discharge_soc_sweep")

sys.path.insert(0, PROJECT_DIR)                   # for the root ml_pipeline.py
sys.path.insert(0, CONTINUOUS_EXPERIMENT_DIR)      # for feature_engineering_continuous.py
sys.path.insert(0, EXP07_DIR)                      # for parse_real_lfp.py / parse_real_nmc.py

from ml_pipeline import run_ml_pipeline
from feature_engineering_continuous import create_features_by_voltage_bins_continuous
from parse_real_lfp import parse_lfp_discharge_files
from parse_real_nmc import parse_nmc_files

RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

SOC_INTERVAL_DIRS = [
    "03_continuous_soc_0.7-1.0", "03_continuous_soc_0.5-0.8",
    "03_continuous_soc_0.3-0.6", "03_continuous_soc_0.1-0.4",
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

    print("\nExtracting features (continuous voltage-bin dV/dQ, >=20% mutual-coverage filter)...")
    features_df = create_features_by_voltage_bins_continuous(RAW_CSV, output_dir=FEATURES_DIR)

    raw_df = pd.read_csv(RAW_CSV)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)

    bin_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    print(f"\nSurviving bins: {bin_cols}")
    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))
    return features_df, bin_cols


def main():
    features_df, bin_cols = extract_features_with_kind()

    synthetic_csv = os.path.join(FEATURES_DIR, "synthetic_features.csv")
    real_csv = os.path.join(FEATURES_DIR, "real_features.csv")
    features_df[features_df['DataKind'] == 'synthetic'].drop(columns=['DataKind']).to_csv(synthetic_csv, index=False)
    features_df[features_df['DataKind'] == 'real'].drop(columns=['DataKind']).to_csv(real_csv, index=False)

    print("\nTraining on synthetic data, testing on real combined data (sim-to-real)...")
    results = run_ml_pipeline(synthetic_csv=synthetic_csv, real_csv=real_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    print("\n=== SUMMARY ===")
    print(f"Surviving bins: {bin_cols}")
    print(f"Accuracies: {results['model_accuracies']}")
    print(f"Top features: {sorted(results['feature_importances'].items(), key=lambda kv: kv[1], reverse=True)[:5]}")


if __name__ == "__main__":
    main()
