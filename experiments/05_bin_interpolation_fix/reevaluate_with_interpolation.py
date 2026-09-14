"""
Re-evaluates the pulse-variable-discharge-rate SOC sweep
(experiments/04_pulse_variable_discharge_soc_sweep/) using the
interpolation-fixed feature engineering in this folder
(feature_engineering_interpolated.py) instead of the live
feature_engineering.py, to test whether interpolating dV/dQ across
skipped bins (see this folder's explanation of the fix) recovers the
signal that collapsed to zero surviving bins in 3 of the 4 SOC intervals.

No re-simulation needed: this reuses the exact raw CSVs already generated
under data/pulse_variable_soc_<interval>/raw/ by experiment 04. Feature
CSVs from this re-evaluation go to a sibling
data/pulse_variable_soc_<interval>/features_interpolated/ folder so they
don't overwrite experiment 04's original (non-interpolated) features.
ml_pipeline.py is reused unchanged from the repo root.

Usage: python3 experiments/05_bin_interpolation_fix/reevaluate_with_interpolation.py
"""
import matplotlib
matplotlib.use("Agg")

import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from ml_pipeline import run_ml_pipeline
from feature_engineering_interpolated import create_features_by_voltage_bins_interpolated

DATA_DIR = os.path.join(PROJECT_DIR, "data")
STEP_COUNT_FOR_TRAINING = 25

# Same 4 intervals experiment 04 already simulated raw data for.
SOC_LABELS = ["0.7-1.0", "0.5-0.8", "0.3-0.6", "0.1-0.4"]

RESULTS_CSV = os.path.join(SCRIPT_DIR, "interpolation_reevaluation_results.csv")
RESULTS_COLUMNS = ["SOC_Interval", "RF_Accuracy", "XGB_Accuracy", "Surviving_Bins", "Top_Features"]


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


def run_one_interval(label):
    run_dir = os.path.join(DATA_DIR, f"pulse_variable_soc_{label}")
    data_csv = os.path.join(run_dir, "raw", "advanced_synthetic_battery_data.csv")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(
            f"{data_csv} not found -- run experiments/04_pulse_variable_discharge_soc_sweep/"
            "sweep_pulse_variable_discharge.py first to generate the raw data."
        )

    print(f"\n{'=' * 70}\nSOC interval {label} (interpolated bins)\n{'=' * 70}")

    features_dir = os.path.join(run_dir, "features_interpolated")
    datasets = create_features_by_voltage_bins_interpolated(
        data_csv, step_counts=[STEP_COUNT_FOR_TRAINING], output_dir=features_dir
    )
    step_subset = datasets[STEP_COUNT_FOR_TRAINING]
    n_batteries = len(step_subset)
    surviving_bins = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]
    print(f"  -> {n_batteries} batteries, surviving bins: {surviving_bins}")

    features_csv = os.path.join(features_dir, f"ml_features_interpolated_soc_{label}_{STEP_COUNT_FOR_TRAINING}_steps.csv")
    step_subset.to_csv(features_csv, index=False)

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

    rf_str = f"{rf_acc:.4f}" if rf_acc is not None else "n/a"
    xgb_str = f"{xgb_acc:.4f}" if xgb_acc is not None else "n/a"
    print(f"Interval {label} done: RF={rf_str} XGB={xgb_str} ({n_batteries} batteries)")
    return row


def main():
    if os.path.exists(RESULTS_CSV):
        print(f"Note: {RESULTS_CSV} already exists -- new rows will be appended, not overwritten.")

    for label in SOC_LABELS:
        try:
            run_one_interval(label)
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

    print(f"\nRe-evaluation complete. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
