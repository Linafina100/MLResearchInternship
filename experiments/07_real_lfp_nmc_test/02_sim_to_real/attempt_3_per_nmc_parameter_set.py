"""
Improvement attempt #3 for experiment 07's sim-to-real test. Attempts #1
(termination-artifact fix) and #2 (broader C-rate diversity) both failed
to close the systematic ~5-10x synthetic-vs-real NMC dV/dQ magnitude gap,
pointing by elimination at a genuine PyBaMM NMC parameter-set mismatch.
Root simulate_batteries.py draws each NMC battery's parameter set
uniformly from NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020",
"OKane2022"] and records it in the raw data's `Base_Parameter_Set`
column -- so far, all three have been pooled together in training. This
attempt tests each set individually against the real NMC data, to see
whether one is meaningfully closer than the pooled average (which could
be masking a partially-working set) or all three are similarly
mismatched.

Reuses experiment 03's already-simulated raw data (no re-simulation),
filtering NMC rows to one Base_Parameter_Set at a time while keeping all
LFP rows (only one parameter set, Prada2013, exists for LFP). Same
artifact fix, real-data parsing, and combined-raw-then-extract-once
pattern as attempts #1/#2.

Usage: python3 experiments/07_real_lfp_nmc_test/02_sim_to_real/attempt_3_per_nmc_parameter_set.py
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

WORK_DIR = os.path.join(SCRIPT_DIR, "per_nmc_parameter_set")
os.makedirs(WORK_DIR, exist_ok=True)
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

SOC_INTERVAL_DIRS = [
    "continuous_soc_0.7-1.0", "continuous_soc_0.5-0.8",
    "continuous_soc_0.3-0.6", "continuous_soc_0.1-0.4",
]
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]


def load_all_synthetic_raw():
    print("Loading experiment 03's already-simulated data (4 SOC intervals, no re-simulation)...")
    dfs = []
    for interval_dir in SOC_INTERVAL_DIRS:
        path = os.path.join(PROJECT_DIR, "data", interval_dir, "raw", "advanced_synthetic_battery_data.csv")
        df = pd.read_csv(path)
        print(f"  -> {interval_dir}: {len(df)} rows")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_real_raw():
    print("\nParsing real LFP data...")
    lfp_df = parse_lfp_discharge_files()
    print("\nParsing real NMC data...")
    nmc_df = parse_nmc_files()
    return pd.concat([lfp_df, nmc_df], ignore_index=True)


def run_for_parameter_set(param_set_name, all_synthetic_df, real_df):
    print(f"\n{'=' * 70}\nNMC parameter set: {param_set_name}\n{'=' * 70}")

    keep = (all_synthetic_df['Chemistry'] == 'LFP') | (all_synthetic_df['Base_Parameter_Set'] == param_set_name)
    sim_df = all_synthetic_df[keep].copy()
    sim_df["DataKind"] = "synthetic"
    n_nmc_batteries_approx = (sim_df['Chemistry'] == 'NMC').sum()
    print(f"  LFP rows: {(sim_df['Chemistry'] == 'LFP').sum()}, NMC ({param_set_name}) rows: {n_nmc_batteries_approx}")

    real_tagged = real_df.copy()
    real_tagged["DataKind"] = "real"

    combined = pd.concat([sim_df, real_tagged], ignore_index=True)
    raw_csv = os.path.join(WORK_DIR, f"raw_{param_set_name}.csv")
    combined.to_csv(raw_csv, index=False)

    features_dir = os.path.join(WORK_DIR, f"features_{param_set_name}")
    features_df = create_features_by_voltage_bins(raw_csv, output_dir=features_dir, exclude_final_transition=True)

    raw_reload = pd.read_csv(raw_csv)
    raw_reload['Battery_ID'] = raw_reload.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_reload.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)

    bin_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    print(f"  Surviving bins: {bin_cols}")

    real_nmc_mean = features_df[(features_df['DataKind'] == 'real') & (features_df['Chemistry'] == 'NMC')][bin_cols].mean()
    sim_nmc_mean = features_df[(features_df['DataKind'] == 'synthetic') & (features_df['Chemistry'] == 'NMC')][bin_cols].mean()
    print(f"  REAL  NMC mean: {real_nmc_mean.round(2).to_dict()}")
    print(f"  SYNTH NMC mean ({param_set_name}): {sim_nmc_mean.round(2).to_dict()}")

    synthetic_csv = os.path.join(features_dir, "synthetic_features.csv")
    real_csv_path = os.path.join(features_dir, "real_features.csv")
    features_df[features_df['DataKind'] == 'synthetic'].drop(columns=['DataKind']).to_csv(synthetic_csv, index=False)
    features_df[features_df['DataKind'] == 'real'].drop(columns=['DataKind']).to_csv(real_csv_path, index=False)

    results = run_ml_pipeline(synthetic_csv=synthetic_csv, real_csv=real_csv_path, save_artifacts=False)
    matplotlib.pyplot.close("all")

    print(f"  Accuracies ({param_set_name}): {results['model_accuracies']}")
    return {
        "param_set": param_set_name,
        "accuracies": results['model_accuracies'],
        "real_nmc_mean": real_nmc_mean.to_dict(),
        "synth_nmc_mean": sim_nmc_mean.to_dict(),
    }


def main():
    all_synthetic_df = load_all_synthetic_raw()
    real_df = load_real_raw()

    all_results = []
    for param_set_name in NMC_PARAMETER_SETS:
        result = run_for_parameter_set(param_set_name, all_synthetic_df, real_df)
        all_results.append(result)

    print(f"\n{'=' * 70}\nSUMMARY: sim-to-real accuracy per NMC parameter set\n{'=' * 70}")
    for r in all_results:
        print(f"{r['param_set']:<12} {r['accuracies']}")


if __name__ == "__main__":
    main()
