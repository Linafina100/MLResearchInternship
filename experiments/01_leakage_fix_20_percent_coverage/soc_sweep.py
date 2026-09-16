"""
Re-run of the SOC-availability sweep (originally run_soc_sweep.py, since
deleted from the repo) using the leakage-fixed feature engineering
documented in this same folder (feature_engineering.py's
min_chemistry_coverage filter -- see RESULTS.md).

Motivation: the original sweep found accuracy was worst at the high-SOC
interval and near-perfect elsewhere, but that was measured on voltage-bin
features that included several bins only one chemistry ever populates --
SimpleImputer's median-fill turned those into a near-perfect chemistry
indicator (see RESULTS.md). Since how much of the voltage range a SOC
interval samples directly affects how many of those bins get populated,
the original sweep's conclusions are suspect. This re-run uses the fixed
feature engineering (which already drops any bin below 20% real-data
coverage for either chemistry by default) to see whether the honest,
much-smaller signal holds up, strengthens, or collapses across SOC ranges.

Each interval runs simulate_batteries.py in a fresh subprocess (so its
module-level state, including the fixed RANDOM_SEED, resets cleanly every
time) and appends one row to RESULTS_CSV immediately -- so a crash partway
through a later interval does not lose earlier results. Raw/feature CSVs
per interval are kept under the repo's data/soc_<interval>/ tree (gitignored,
~200MB each), not duplicated into this experiments folder -- only the small
results summary lives here.

Usage: python3 experiments/01_leakage_fix_20_percent_coverage/soc_sweep.py
"""
import matplotlib
matplotlib.use("Agg")  # must happen before any of the imports below pull in pyplot

import csv
import os
import subprocess
import sys
import time

# This script lives at <repo_root>/experiments/01_.../soc_sweep.py; walk up
# three levels to find the repo root regardless of cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# Continuous discharge was promoted to the root simulate_batteries.py/
# feature_engineering.py; this sweep specifically needs the GITT pulse
# protocol, archived at experiments/07_pulse_protocol_archive/ -- inserted
# last (so first in sys.path) to shadow the root feature_engineering.py.
PULSE_ARCHIVE_DIR = os.path.join(PROJECT_DIR, "experiments", "07_pulse_protocol_archive")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, PULSE_ARCHIVE_DIR)

from feature_engineering import create_features_by_voltage_bins
from ml_pipeline import run_ml_pipeline

SIMULATE_SCRIPT = os.path.join(PULSE_ARCHIVE_DIR, "simulate_batteries.py")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# (soc_max, soc_min) pairs -- matches simulate_batteries.py's
# random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX) draw for starting SOC.
SOC_INTERVALS = [(1.0, 0.7), (0.8, 0.5), (0.6, 0.3), (0.4, 0.1)]

# All prior accuracy numbers reported in this project used the 25-step
# dataset; kept consistent here so sweep results are directly comparable.
STEP_COUNT_FOR_TRAINING = 25

RESULTS_CSV = os.path.join(SCRIPT_DIR, "soc_sweep_results.csv")
RESULTS_COLUMNS = ["SOC_Interval", "RF_Accuracy", "XGB_Accuracy", "Surviving_Bins", "Top_Features"]


def _interval_label(soc_max, soc_min):
    return f"{soc_min:.1f}-{soc_max:.1f}"


def _format_top_features(feature_importances, n=3):
    ranked = sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return "; ".join(f"{name}:{importance:.4f}" for name, importance in ranked)


def _append_result_row(row):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one_interval(soc_max, soc_min):
    label = _interval_label(soc_max, soc_min)
    print(f"\n{'=' * 70}\nSOC interval {label} "
          f"(SOC_RANGE_MIN={soc_min}, SOC_RANGE_MAX={soc_max})\n{'=' * 70}")
    t_start = time.time()

    run_label = f"soc_{label}"
    run_dir = os.path.join(DATA_DIR, run_label)
    data_csv = os.path.join(run_dir, "raw", "advanced_synthetic_battery_data.csv")

    env = os.environ.copy()
    env["SOC_RANGE_MIN"] = str(soc_min)
    env["SOC_RANGE_MAX"] = str(soc_max)
    env["RUN_LABEL"] = run_label
    env.setdefault("MPLBACKEND", "Agg")

    print("[1/3] Simulating (slow step, ~10 min)...")
    subprocess.run([sys.executable, SIMULATE_SCRIPT], cwd=PROJECT_DIR, env=env, check=True)

    print("[2/3] Feature engineering (voltage bins, >=20% mutual-coverage filter applied)...")
    features_dir = os.path.join(run_dir, "features")
    datasets = create_features_by_voltage_bins(
        data_csv, step_counts=[STEP_COUNT_FOR_TRAINING], output_dir=features_dir
    )
    step_subset = datasets[STEP_COUNT_FOR_TRAINING]
    n_batteries = len(step_subset)
    surviving_bins = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]
    print(f"  -> {n_batteries} batteries, surviving bins: {surviving_bins}")

    features_csv = os.path.join(features_dir, f"ml_features_soc_{label}_{STEP_COUNT_FOR_TRAINING}_steps.csv")
    step_subset.to_csv(features_csv, index=False)

    print("[3/3] Training (Random Forest + XGBoost, artifacts not saved)...")
    results = run_ml_pipeline(synthetic_csv=features_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")  # run_ml_pipeline opens a figure per call

    rf_acc = results["model_accuracies"].get("Random Forest")
    xgb_acc = results["model_accuracies"].get("XGBoost")
    top_features = _format_top_features(results["feature_importances"])

    row = {
        "SOC_Interval": label,
        "RF_Accuracy": f"{rf_acc:.4f}" if rf_acc is not None else "",
        "XGB_Accuracy": f"{xgb_acc:.4f}" if xgb_acc is not None else "",
        "Surviving_Bins": "; ".join(surviving_bins),
        "Top_Features": top_features,
    }
    _append_result_row(row)

    elapsed_min = (time.time() - t_start) / 60
    print(f"Interval {label} done in {elapsed_min:.1f} min: "
          f"RF={rf_acc:.4f} XGB={xgb_acc:.4f} ({n_batteries} batteries)")
    return row


def main():
    if os.path.exists(RESULTS_CSV):
        print(f"Note: {RESULTS_CSV} already exists -- new rows will be appended, not overwritten.")

    sweep_start = time.time()
    for soc_max, soc_min in SOC_INTERVALS:
        label = _interval_label(soc_max, soc_min)
        try:
            run_one_interval(soc_max, soc_min)
        except Exception as exc:
            print(f"!! Interval {label} FAILED: {type(exc).__name__}: {exc}")
            _append_result_row({
                "SOC_Interval": label,
                "RF_Accuracy": "ERROR",
                "XGB_Accuracy": "ERROR",
                "Surviving_Bins": "",
                "Top_Features": f"{type(exc).__name__}: {exc}",
            })
            continue

    total_min = (time.time() - sweep_start) / 60
    print(f"\nSweep complete in {total_min:.1f} min. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
