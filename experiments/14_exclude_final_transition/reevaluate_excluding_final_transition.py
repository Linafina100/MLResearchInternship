"""
Alternative fix for the termination-step artifact (see
experiments/11_voltage_cutoff_comparison and
experiments/12_exclude_termination_artifact_bins), tried instead of a
simulation-level resampling fix that turned out to make things worse (see
experiments/13_finer_discharge_sampling -- ruled out).

Experiment 12 excludes whole bins within one bin-width of either
chemistry's cutoff. This is coarser than necessary: only each battery's
*single final* raw transition is the artifact (an oversized, irregular
last step landing on the voltage cutoff -- see experiment 11's evidence).
Root feature_engineering.py now supports an `exclude_final_transition`
flag that drops just that one transition per battery before binning,
rather than dropping entire bins after the fact. The hope: bins that were
only ever populated by the artifact should then naturally fail the
existing `>=20%` mutual-coverage filter on their own (no artifact
transition left to fill them), while bins with genuine mid-trace coverage
from both chemistries -- including ones close to the cutoff that
experiment 12's blunter bin-width rule might have dropped unnecessarily --
are preserved.

No re-simulation needed -- reuses experiment 03's already-committed raw
CSVs (data/continuous_soc_<interval>/raw/), and root
feature_engineering.py / ml_pipeline.py otherwise unchanged, same
reevaluate-existing-data pattern as experiments 05 and 12.

Usage: python3 experiments/14_exclude_final_transition/reevaluate_excluding_final_transition.py
"""
import matplotlib
matplotlib.use("Agg")  # must happen before any of the imports below pull in pyplot

import csv
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)  # for root feature_engineering.py and ml_pipeline.py

from feature_engineering import create_features_by_voltage_bins
from ml_pipeline import run_ml_pipeline

DATA_DIR = os.path.join(PROJECT_DIR, "data")

SOC_INTERVALS = [(1.0, 0.7), (0.8, 0.5), (0.6, 0.3), (0.4, 0.1)]  # same 4 as exp03

RESULTS_CSV = os.path.join(SCRIPT_DIR, "final_transition_exclusion_results.csv")
RESULTS_COLUMNS = [
    "SOC_Interval", "Bin_Set",
    "RF_Accuracy", "XGB_Accuracy", "Surviving_Bins", "Top_Features",
]


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


def run_one(label, bin_set_name, features_df, features_csv):
    print(f"[{label}/{bin_set_name}] training on {[c for c in features_df.columns if c.startswith('dV_dQ_V_')]}...")
    features_df.to_csv(features_csv, index=False)

    results = run_ml_pipeline(synthetic_csv=features_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    rf_acc = results["model_accuracies"].get("Random Forest")
    xgb_acc = results["model_accuracies"].get("XGBoost")
    surviving_bins = [c for c in features_df.columns if c.startswith("dV_dQ_V_")]
    top_features = _format_top_features(results["feature_importances"])

    row = {
        "SOC_Interval": label,
        "Bin_Set": bin_set_name,
        "RF_Accuracy": f"{rf_acc:.4f}" if rf_acc is not None else "",
        "XGB_Accuracy": f"{xgb_acc:.4f}" if xgb_acc is not None else "",
        "Surviving_Bins": "; ".join(surviving_bins),
        "Top_Features": top_features,
    }
    _append_result_row(row)

    rf_str = f"{rf_acc:.4f}" if rf_acc is not None else "n/a"
    xgb_str = f"{xgb_acc:.4f}" if xgb_acc is not None else "n/a"
    print(f"  -> RF={rf_str} XGB={xgb_str} ({len(surviving_bins)} bins)")
    return row


def run_one_interval(soc_max, soc_min):
    label = _interval_label(soc_max, soc_min)
    print(f"\n{'=' * 70}\nSOC interval {label}\n{'=' * 70}")

    run_dir = os.path.join(DATA_DIR, f"continuous_soc_{label}")
    data_csv = os.path.join(run_dir, "raw", "advanced_synthetic_battery_data.csv")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(
            f"{data_csv} not found -- this script reuses experiment 03's raw data "
            "and does not re-simulate. Run experiment 03's sweep first."
        )

    features_dir = os.path.join(SCRIPT_DIR, "features")

    before_df = create_features_by_voltage_bins(data_csv, output_dir=features_dir, exclude_final_transition=False)
    before_csv = os.path.join(features_dir, f"ml_features_before_soc_{label}.csv")
    run_one(label, "before_exclusion", before_df, before_csv)

    after_df = create_features_by_voltage_bins(data_csv, output_dir=features_dir, exclude_final_transition=True)
    after_csv = os.path.join(features_dir, f"ml_features_after_soc_{label}.csv")
    run_one(label, "after_exclusion", after_df, after_csv)


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
            for bin_set_name in ("before_exclusion", "after_exclusion"):
                _append_result_row({
                    "SOC_Interval": label,
                    "Bin_Set": bin_set_name,
                    "RF_Accuracy": "ERROR",
                    "XGB_Accuracy": "ERROR",
                    "Surviving_Bins": "",
                    "Top_Features": f"{type(exc).__name__}: {exc}",
                })
                continue

    total_min = (time.time() - sweep_start) / 60
    print(f"\nDone in {total_min:.1f} min. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
