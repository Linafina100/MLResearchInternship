"""
Phase 2 of experiment 16: sim-to-real test restricted to the real-world
target zone (2.5-3.0V), using synthetic data simulated at an artificially
lowered cutoff (1.5V for both chemistries -- see diagnose_lower_cutoff.py
and RESULTS.md's Phase 1) so the solver-termination artifact lands well
below the target zone instead of contaminating it.

Data sources:
- Synthetic: data/low_voltage_v1.5/raw/advanced_synthetic_battery_data.csv,
  produced by running root simulate_batteries.py with
  LFP_LOWER_CUTOFF=1.5 NMC_LOWER_CUTOFF=1.5 RUN_LABEL=low_voltage_v1.5
  (unmodified script, env-var override only -- see RESULTS.md).
- Real: experiment 07's own parsers (parse_real_lfp.py, parse_real_nmc.py),
  unchanged.

Method: combine synthetic + real, extract features ONCE with root
feature_engineering.py (exclude_final_transition=True kept as an extra
safety net) so both sides share identical bin columns, then restrict the
feature columns used for training/testing to only those inside the
2.5-3.0V target zone -- done here, not by adding a new parameter to
feature_engineering.py, since this is a single-experiment need (see the
discarded outlier-jump-multiplier attempt in experiment 07 for why
shared-code changes for a single experiment's need are avoided).

Usage: python3 experiments/16_low_voltage_only_sim_to_real/evaluate_low_voltage_sim_to_real.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import re
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXP07_DIR = os.path.join(PROJECT_DIR, "experiments", "07_real_lfp_nmc_test")

sys.path.insert(0, PROJECT_DIR)   # for root ml_pipeline.py and feature_engineering.py
sys.path.insert(0, EXP07_DIR)     # for parse_real_lfp.py / parse_real_nmc.py

from ml_pipeline import run_ml_pipeline
from feature_engineering import create_features_by_voltage_bins
from parse_real_lfp import parse_lfp_discharge_files
from parse_real_nmc import parse_nmc_files

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "low_voltage_v1.5", "raw", "advanced_synthetic_battery_data.csv")
RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
BIN_COL_RE = re.compile(r"dV_dQ_V_([\d.]+)_([\d.]+)")


def build_combined_raw_csv():
    print(f"Loading experiment 16's low-cutoff (1.5V) synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"  -> {len(sim_df)} rows")

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


def target_zone_bin_cols(all_bin_cols):
    kept = []
    for col in all_bin_cols:
        m = BIN_COL_RE.match(col)
        bin_high, bin_low = float(m.group(1)), float(m.group(2))
        if bin_low >= TARGET_ZONE_MIN and bin_high <= TARGET_ZONE_MAX:
            kept.append(col)
    return kept


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

    all_bin_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    zone_bin_cols = target_zone_bin_cols(all_bin_cols)
    print(f"\nAll surviving bins: {all_bin_cols}")
    print(f"Target-zone ({TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V) bins kept for training: {zone_bin_cols}")
    dropped_cols = [c for c in all_bin_cols if c not in zone_bin_cols]
    if dropped_cols:
        print(f"Dropped (outside target zone): {dropped_cols}")

    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))
    return features_df, zone_bin_cols


def main():
    features_df, zone_bin_cols = extract_features_with_kind()

    if not zone_bin_cols:
        print("\nNo surviving bins inside the target zone -- cannot train. Stopping here.")
        return

    metadata_cols = [
        'Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC',
        'Target_Capacity_Ah', 'Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set',
    ]
    keep_cols = [c for c in metadata_cols if c in features_df.columns] + zone_bin_cols

    synthetic_csv = os.path.join(FEATURES_DIR, "synthetic_features_target_zone.csv")
    real_csv = os.path.join(FEATURES_DIR, "real_features_target_zone.csv")
    features_df[features_df['DataKind'] == 'synthetic'][keep_cols].to_csv(synthetic_csv, index=False)
    features_df[features_df['DataKind'] == 'real'][keep_cols].to_csv(real_csv, index=False)

    print(f"\nTraining on synthetic data (1.5V cutoff), testing on real data -- "
          f"features restricted to the {TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V target zone only...")
    results = run_ml_pipeline(synthetic_csv=synthetic_csv, real_csv=real_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    print("\n=== SUMMARY ===")
    print(f"Target-zone bins used: {zone_bin_cols}")
    print(f"Accuracies: {results['model_accuracies']}")
    print(f"Top features: {sorted(results['feature_importances'].items(), key=lambda kv: kv[1], reverse=True)[:5]}")
    print("\nFor context, experiment 07's full-curve sim-to-real baseline scored 24.84% "
          "(both models); its fragile per-NMC-parameter-set best case scored ~84% RF "
          "(shown not to survive pooling -- see ../07_real_lfp_nmc_test/02_sim_to_real/RESULTS.md).")


if __name__ == "__main__":
    main()
