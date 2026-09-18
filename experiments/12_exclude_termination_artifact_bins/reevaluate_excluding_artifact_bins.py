"""
Quantifies how much of experiment 03's low-SOC accuracy survives once the
solver-termination artifact bins are excluded.

Background (full evidence in experiments/11_voltage_cutoff_comparison/RESULTS.md):
PyBaMM's "Discharge until
X V" experiment is event-triggered, so every battery's last raw sample
lands within 0.02V of its chemistry's voltage cutoff via one oversized
final step. Dividing dV by the resulting near-zero final dQ explodes the
dV/dQ value in whichever bin sits right at that cutoff. In experiment 03
(cutoffs LFP=1.8V, NMC=2.3V), this shows up as `dV_dQ_V_2.4_2.3` and
`dV_dQ_V_2.3_2.2` -- both right at NMC's cutoff -- surviving the mutual-
coverage filter at the two lowest SOC intervals, where NMC's ~500-2000x
larger-magnitude artifact values sit alongside LFP's normal ones in the
same bin, giving the classifier a trivial (non-electrochemical) tell.

Fix applied here, generalized rather than hardcoded to those two exact
bin names: after root feature_engineering.py produces its normal set of
coverage-filtered bins, drop any bin within one bin-width (0.1V) of
*either* chemistry's cutoff (so both the exact boundary bin and its
immediate neighbor, to absorb the +-0.02V measurement noise observed
around the cutoff). No re-simulation needed -- this reuses experiment 03's
already-committed raw CSVs (data/continuous_soc_<interval>/raw/), reusing
root feature_engineering.py and ml_pipeline.py unchanged, same pattern as
experiment 05's before/after reevaluation of experiment 04's data.

Usage: python3 experiments/12_exclude_termination_artifact_bins/reevaluate_excluding_artifact_bins.py
"""
import matplotlib
matplotlib.use("Agg")  # must happen before any of the imports below pull in pyplot

import csv
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)  # for root feature_engineering.py and ml_pipeline.py

from feature_engineering import create_features_by_voltage_bins
from ml_pipeline import run_ml_pipeline

DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Same cutoffs experiment 03 used (root's defaults, unchanged on this branch).
LOWER_VOLTAGE_CUTOFF = {"LFP": 1.8, "NMC": 2.3}
V_BIN_WIDTH = 0.1

SOC_INTERVALS = [(1.0, 0.7), (0.8, 0.5), (0.6, 0.3), (0.4, 0.1)]  # same 4 as exp03

RESULTS_CSV = os.path.join(SCRIPT_DIR, "artifact_bin_exclusion_results.csv")
RESULTS_COLUMNS = [
    "SOC_Interval", "Bin_Set",
    "RF_Accuracy", "XGB_Accuracy", "Surviving_Bins", "Top_Features",
]

BIN_NAME_RE = re.compile(r"^dV_dQ_V_([0-9.]+)_([0-9.]+)$")


def _is_artifact_adjacent(bin_col):
    """True if this bin's upper edge sits within one bin-width of either
    chemistry's voltage cutoff (the boundary the termination-step
    artifact lands on/near)."""
    m = BIN_NAME_RE.match(bin_col)
    if not m:
        return False
    bin_high = float(m.group(1))
    return any(abs(bin_high - cutoff) <= V_BIN_WIDTH + 1e-9 for cutoff in LOWER_VOLTAGE_CUTOFF.values())


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
    full_df = create_features_by_voltage_bins(data_csv, output_dir=features_dir)

    bin_cols = [c for c in full_df.columns if c.startswith("dV_dQ_V_")]
    artifact_cols = [c for c in bin_cols if _is_artifact_adjacent(c)]
    print(f"  Bins before exclusion: {bin_cols}")
    print(f"  Artifact-adjacent bins to exclude: {artifact_cols}")

    before_csv = os.path.join(features_dir, f"ml_features_before_soc_{label}.csv")
    run_one(label, "before_exclusion", full_df, before_csv)

    filtered_df = full_df.drop(columns=artifact_cols)
    after_csv = os.path.join(features_dir, f"ml_features_after_soc_{label}.csv")
    run_one(label, "after_exclusion", filtered_df, after_csv)


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
