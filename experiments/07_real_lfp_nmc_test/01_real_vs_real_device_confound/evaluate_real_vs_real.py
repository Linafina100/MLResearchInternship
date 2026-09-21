"""
Original design for experiment 07: combines the real LFP
(parse_real_lfp.py) and real NMC (parse_real_nmc.py) data into one raw
dataset, extracts features with
experiments/03_continuous_discharge_soc_sweep/feature_engineering_continuous.py
(unchanged), and evaluates classification with ml_pipeline.py (unchanged,
repo root) -- a plain train/test split on real, combined data, not a
sim-to-real transfer test.

Result (see ../RESULTS.md): 100% accuracy for both Random Forest and
XGBoost. sibling script evaluate_device_id_control.py runs the control
that shows why this number isn't a real chemistry-classification
finding -- it relabels the 3 real NMC cells by device identity instead of
chemistry and gets the same 100%, meaning the classifier is picking up
per-device/per-lab measurement fingerprints, not LFP-vs-NMC signal.

Usage: python3 experiments/07_real_lfp_nmc_test/01_real_vs_real_device_confound/evaluate_real_vs_real.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

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

RAW_CSV = os.path.join(SCRIPT_DIR, "real_combined_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")


def build_combined_raw_csv():
    lfp_df = parse_lfp_discharge_files()
    nmc_df = parse_nmc_files()
    combined = __import__("pandas").concat([lfp_df, nmc_df], ignore_index=True)
    combined.to_csv(RAW_CSV, index=False)
    print(f"\nCombined raw data saved to '{RAW_CSV}': "
          f"{(combined['Chemistry'] == 'LFP').sum()} LFP rows, "
          f"{(combined['Chemistry'] == 'NMC').sum()} NMC rows")
    return combined


def main():
    if not os.path.exists(RAW_CSV):
        build_combined_raw_csv()
    else:
        print(f"Note: '{RAW_CSV}' already exists -- reusing it. Delete it to re-parse from source.")

    print("\nExtracting features (continuous voltage-bin dV/dQ, >=20% mutual-coverage filter)...")
    features_df = create_features_by_voltage_bins_continuous(RAW_CSV, output_dir=FEATURES_DIR)
    n_batteries = len(features_df)
    surviving_bins = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    print(f"  -> {n_batteries} real batteries (cycles), surviving bins: {surviving_bins}")

    features_csv = os.path.join(FEATURES_DIR, "ml_features_real_combined.csv")
    features_df.to_csv(features_csv, index=False)

    print("\nTraining (Random Forest + XGBoost) on real combined data...")
    results = run_ml_pipeline(synthetic_csv=features_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    print("\n=== SUMMARY ===")
    print(f"Batteries (cycles): {n_batteries}")
    print(f"Surviving bins: {surviving_bins}")
    print(f"Accuracies: {results['model_accuracies']}")
    print(f"Top features: {sorted(results['feature_importances'].items(), key=lambda kv: kv[1], reverse=True)[:5]}")


if __name__ == "__main__":
    main()
