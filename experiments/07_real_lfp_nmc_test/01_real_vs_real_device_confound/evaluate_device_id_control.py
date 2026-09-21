"""
Control test for evaluate_real_vs_real.py's 100% LFP-vs-NMC accuracy.

parse_real_nmc.py's real NMC data comes from 3 distinct physical cells
tested at different temperatures (CY25-05_1-#1, CY35-05_1-#1,
CY45-05_1-#1 -- see ../parse_real_nmc.py), each with its own
Variation_ID prefix ("<cell_label>_cycle_<n>"). This script keeps only
those NMC rows, relabels the `Chemistry` column with the cell's device
identity (CY25 / CY35 / CY45) instead of the chemistry name, and runs the
exact same feature-extraction + ml_pipeline classification used for the
real chemistry test.

If a classifier can tell these 3 same-chemistry cells apart from dV/dQ
features alone just as well as it told LFP from NMC apart, that's direct
evidence the original 100% accuracy reflects per-device/per-lab
measurement fingerprints (sampling rate, equipment noise floor, protocol
quirks) rather than genuine LFP-vs-NMC chemistry signal -- since here
there is no chemistry difference to detect at all, only device identity.

No historical script for this control existed in the repo (the original
experiment 07 RESULTS.md describes it as having been run ad hoc); this
is a fresh implementation written to reproduce and verify that claim.

Usage: python3 experiments/07_real_lfp_nmc_test/01_real_vs_real_device_confound/evaluate_device_id_control.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP07_DIR = os.path.dirname(SCRIPT_DIR)  # experiments/07_real_lfp_nmc_test/, for the shared parser
PROJECT_DIR = os.path.dirname(os.path.dirname(EXP07_DIR))
CONTINUOUS_EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments", "03_continuous_discharge_soc_sweep")

sys.path.insert(0, PROJECT_DIR)                   # for the root ml_pipeline.py
sys.path.insert(0, CONTINUOUS_EXPERIMENT_DIR)      # for feature_engineering_continuous.py
sys.path.insert(0, EXP07_DIR)                      # for parse_real_nmc.py

from ml_pipeline import run_ml_pipeline
from feature_engineering_continuous import create_features_by_voltage_bins_continuous
from parse_real_nmc import parse_nmc_files

RAW_CSV = os.path.join(SCRIPT_DIR, "nmc_device_id_raw.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features_device_id")


def build_device_labeled_raw_csv():
    nmc_df = parse_nmc_files()
    # Variation_ID is "<cell_label>_cycle_<n>" (see parse_real_nmc.py);
    # the cell_label (e.g. "CY25-05_1-#1") is the device identity.
    nmc_df = nmc_df.copy()
    nmc_df["Chemistry"] = nmc_df["Variation_ID"].str.split("_cycle_").str[0]
    nmc_df.to_csv(RAW_CSV, index=False)
    print(f"\nDevice-labeled raw data saved to '{RAW_CSV}': "
          f"{nmc_df['Chemistry'].value_counts().to_dict()} rows per device")
    return nmc_df


def main():
    if not os.path.exists(RAW_CSV):
        build_device_labeled_raw_csv()
    else:
        print(f"Note: '{RAW_CSV}' already exists -- reusing it. Delete it to re-parse from source.")

    print("\nExtracting features (continuous voltage-bin dV/dQ, >=20% mutual-coverage filter)...")
    features_df = create_features_by_voltage_bins_continuous(RAW_CSV, output_dir=FEATURES_DIR)
    n_batteries = len(features_df)
    surviving_bins = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    print(f"  -> {n_batteries} real NMC cycles across 3 devices, surviving bins: {surviving_bins}")
    print(features_df['Chemistry'].value_counts().rename('n_cycles'))

    features_csv = os.path.join(FEATURES_DIR, "ml_features_device_id.csv")
    features_df.to_csv(features_csv, index=False)

    print("\nTraining (Random Forest + XGBoost) to predict DEVICE IDENTITY "
          "(same chemistry, different physical cells)...")
    results = run_ml_pipeline(synthetic_csv=features_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    print("\n=== SUMMARY ===")
    print(f"Cycles: {n_batteries}")
    print(f"Surviving bins: {surviving_bins}")
    print(f"Accuracies (predicting device ID, not chemistry): {results['model_accuracies']}")
    print(f"Top features: {sorted(results['feature_importances'].items(), key=lambda kv: kv[1], reverse=True)[:5]}")
    print("\nIf this accuracy is comparable to evaluate_real_vs_real.py's LFP-vs-NMC "
          "accuracy, the original result is a same-device/same-lab confound, not a "
          "genuine chemistry-classification finding.")


if __name__ == "__main__":
    main()
