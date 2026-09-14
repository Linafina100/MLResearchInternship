"""
SOC-availability sweep using the continuous-discharge protocol (0.6C, runs
to the voltage cutoff, no pulses/rests) instead of GITT pulses -- see
simulate_batteries_continuous.py and feature_engineering_continuous.py in
this same folder for what changed and why.

Structurally identical to experiments/01_leakage_fix_20_percent_coverage/
soc_sweep.py: same 4 SOC intervals, a fresh subprocess per interval so
RANDOM_SEED resets cleanly, results appended incrementally so a crash
partway through doesn't lose earlier intervals. ml_pipeline.py is reused
unchanged from the repo root (it's feature-set-agnostic), imported
directly rather than copied.

Usage: python3 experiments/03_continuous_discharge_soc_sweep/sweep_continuous_discharge.py
"""
import matplotlib
matplotlib.use("Agg")  # must happen before any of the imports below pull in pyplot

import csv
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)   # for the root ml_pipeline.py
sys.path.insert(0, SCRIPT_DIR)    # for this folder's feature_engineering_continuous.py

from ml_pipeline import run_ml_pipeline
from feature_engineering_continuous import create_features_by_voltage_bins_continuous

SIMULATE_SCRIPT = os.path.join(SCRIPT_DIR, "simulate_batteries_continuous.py")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# (soc_max, soc_min) pairs -- same 4 intervals as the pulse-based sweep.
SOC_INTERVALS = [(1.0, 0.7), (0.8, 0.5), (0.6, 0.3), (0.4, 0.1)]

RESULTS_CSV = os.path.join(SCRIPT_DIR, "continuous_discharge_soc_sweep_results.csv")
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

    run_label = f"continuous_soc_{label}"
    run_dir = os.path.join(DATA_DIR, run_label)
    data_csv = os.path.join(run_dir, "raw", "advanced_synthetic_battery_data.csv")

    env = os.environ.copy()
    env["SOC_RANGE_MIN"] = str(soc_min)
    env["SOC_RANGE_MAX"] = str(soc_max)
    env["RUN_LABEL"] = run_label
    env.setdefault("MPLBACKEND", "Agg")

    print("[1/3] Simulating (continuous discharge, expect faster than the pulse sweep)...")
    subprocess.run([sys.executable, SIMULATE_SCRIPT], cwd=PROJECT_DIR, env=env, check=True)

    print("[2/3] Feature engineering (continuous dV/dQ voltage bins, >=20% mutual-coverage filter)...")
    features_dir = os.path.join(run_dir, "features")
    step_subset = create_features_by_voltage_bins_continuous(data_csv, output_dir=features_dir)
    n_batteries = len(step_subset)
    surviving_bins = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]
    print(f"  -> {n_batteries} batteries, surviving bins: {surviving_bins}")

    features_csv = os.path.join(features_dir, f"ml_features_continuous_soc_{label}.csv")
    step_subset.to_csv(features_csv, index=False)

    print("[3/3] Training (Random Forest + XGBoost, artifacts not saved)...")
    results = run_ml_pipeline(synthetic_csv=features_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

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
    rf_str = f"{rf_acc:.4f}" if rf_acc is not None else "n/a"
    xgb_str = f"{xgb_acc:.4f}" if xgb_acc is not None else "n/a"
    print(f"Interval {label} done in {elapsed_min:.1f} min: "
          f"RF={rf_str} XGB={xgb_str} ({n_batteries} batteries)")
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
