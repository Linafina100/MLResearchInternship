"""
SOC-availability sweep using the continuous-discharge protocol.
Automates simulation, continuous dV/dQ binning, ML training, and saves all
generated plots (discharge curves, dV/dQ profiles, confusion matrices/feature importances)
across 4 distinct SOC intervals.
"""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to ensure clean headless execution

import csv
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)   # for root ml_pipeline.py
sys.path.insert(0, SCRIPT_DIR)    # for local feature engineering

from ML_pipeline_const_random import run_ml_pipeline
from feature_engineering_const_random import (
    create_features_by_voltage_bins_continuous,
    plot_dvdq_profiles,
)

SIMULATE_SCRIPT = os.path.join(SCRIPT_DIR, "simulate_batteries_const_random.py")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# (soc_max, soc_min) evaluation windows
SOC_INTERVALS = [(1.0, 0.7), (0.8, 0.5), (0.6, 0.3), (0.4, 0.1)]
#SOC_INTERVALS = [ (0.4, 0.1)]

RESULTS_CSV = os.path.join(SCRIPT_DIR, "continuous_discharge_soc_sweep_results.csv")
RESULTS_COLUMNS = ["SOC_Interval", "RF_Accuracy", "XGB_Accuracy", "Surviving_Bins", "Top_Features"]


def _interval_label(soc_max, soc_min):
    return f"{soc_min:.1f}-{soc_max:.1f}"


def _format_top_features(feature_importances, n=3):
    if not feature_importances:
        return "None"
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
    print(f"\n{'=' * 70}\nStarting SOC interval {label} "
          f"(SOC_RANGE_MIN={soc_min}, SOC_RANGE_MAX={soc_max})\n{'=' * 70}")
    t_start = time.time()

    run_label = f"continuous_soc_{label}"
    run_dir = os.path.join(DATA_DIR, run_label)
    
    plots_dir = os.path.join(run_dir, "plots")
    models_dir = os.path.join(run_dir, "models")
    features_dir = os.path.join(run_dir, "features")
    raw_dir = os.path.join(run_dir, "raw")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    data_csv = os.path.join(raw_dir, "continuous_synthetic_battery_data.csv")
    features_csv = os.path.join(features_dir, f"ml_features_continuous_soc_{label}.csv")

    # -----------------------------------------------------------------------
    # [1/3] SIMULATION & DISCHARGE PLOT
    # -----------------------------------------------------------------------
    print("[1/3] Simulating continuous discharge runs...")
    discharge_plot_path = os.path.join(plots_dir, f"discharge_curves_soc_{label}.png")

    env = os.environ.copy()
    env["SOC_RANGE_MIN"] = str(soc_min)
    env["SOC_RANGE_MAX"] = str(soc_max)
    env["RUN_LABEL"] = run_label
    env["OUTPUT_DATA_CSV"] = data_csv
    env["DISCHARGE_PLOT_PNG"] = discharge_plot_path
    env["MPLBACKEND"] = "Agg"

    subprocess.run([sys.executable, SIMULATE_SCRIPT], cwd=PROJECT_DIR, env=env, check=True)
    print(f"  -> Raw data written to: '{data_csv}'")
    print(f"  -> Discharge plot saved to: '{discharge_plot_path}'")

    # -----------------------------------------------------------------------
    # [2/3] FEATURE ENGINEERING & dV/dQ PLOT
    # -----------------------------------------------------------------------
    print("[2/3] Extracting continuous dV/dQ voltage bins...")
    step_subset, out_dir = create_features_by_voltage_bins_continuous(data_csv, output_dir=features_dir)
    step_subset.to_csv(features_csv, index=False)

    n_batteries = len(step_subset)
    surviving_bins = [c for c in step_subset.columns if c.startswith("dV_dQ_V_")]
    print(f"  -> {n_batteries} batteries processed, {len(surviving_bins)} active voltage bins")

    # Generate and save dV/dQ profile plot for this interval
    dvdq_plot_filename = f"dvdq_profiles_soc_{label}.png"
    plot_dvdq_profiles(step_subset, output_dir=plots_dir, filename=dvdq_plot_filename)

    # -----------------------------------------------------------------------
    # [3/3] ML TRAINING & EVALUATION PLOTS
    # -----------------------------------------------------------------------
    print("[3/3] Training Random Forest and XGBoost models...")
    ml_plot_filename = f"ml_eval_plots_soc_{label}.png"
    
    results = run_ml_pipeline(
        synthetic_csv=features_csv,
        save_artifacts=True,
        models_dir=models_dir,
        plot_filename=ml_plot_filename,
    )

    rf_acc = results["model_accuracies"].get("Random Forest")
    xgb_acc = results["model_accuracies"].get("XGBoost")
    top_features = _format_top_features(results["feature_importances"])

    # -----------------------------------------------------------------------
    # LOGGING
    # -----------------------------------------------------------------------
    row = {
        "SOC_Interval": label,
        "RF_Accuracy": f"{rf_acc:.4f}" if rf_acc is not None else "",
        "XGB_Accuracy": f"{xgb_acc:.4f}" if xgb_acc is not None else "",
        "Surviving_Bins": "; ".join(surviving_bins),
        "Top_Features": top_features,
    }
    _append_result_row(row)

    elapsed_min = (time.time() - t_start) / 60
    rf_str = f"{rf_acc * 100:.2f}%" if rf_acc is not None else "n/a"
    xgb_str = f"{xgb_acc * 100:.2f}%" if xgb_acc is not None else "n/a"
    print(f"\nInterval {label} completed in {elapsed_min:.1f} min:")
    print(f"  -> Random Forest Accuracy : {rf_str}")
    print(f"  -> XGBoost Accuracy       : {xgb_str}")
    print(f"  -> Plots saved to         : '{plots_dir}' and '{models_dir}'")

    return row


def main():
    if os.path.exists(RESULTS_CSV):
        print(f"Notice: Existing '{RESULTS_CSV}' found. Appending results to existing file.")

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
    print(f"\nAll sweeps finished in {total_min:.1f} minutes.")
    print(f"Aggregated numerical report saved to: '{RESULTS_CSV}'")


if __name__ == "__main__":
    main()