"""
Systematic sweep over starting-SOC intervals, to find where LFP/NMC
classification performance breaks down as high-voltage OCV data disappears.

Motivation: every result so far in this project used SOC randomized over
50-100%, which always gives the model *some* access to the high-SOC region.
End-of-life cells arriving at Stena won't cooperate -- a cell pulled off the
line could be sitting anywhere on its discharge curve, including well below
50%. This sweep re-runs the full pipeline (simulate -> feature-engineer ->
train) across progressively lower SOC windows to see exactly where accuracy
degrades as that "free" high-voltage information is taken away, while
preserving every other realistic-degradation change already made on this
branch (50-85% SOH, decoupled resistance noise, ambient-temperature effect,
NMC parameter-set pooling).

Each interval runs simulate_batteries.py in a fresh subprocess (so its
module-level state, including the fixed RANDOM_SEED, resets cleanly every
time) and appends one row to data/summary/soc_sweep_results.csv immediately
-- so a crash partway through a later interval does not lose earlier
results.

Runtime: PyBaMM solving dominates, ~10-11 minutes per interval, so expect
~40-45 minutes total for all four intervals. Each interval's raw simulation
CSV (~200MB), kept under data/soc_<interval>/raw/, is kept, not deleted, for
later inspection -- expect ~800MB of extra disk usage across all four
intervals.

Usage: python3 run_soc_sweep.py
"""
import matplotlib
matplotlib.use("Agg")  # must happen before any of the imports below pull in pyplot

import csv
import os
import subprocess
import sys
import time

from feature_engineering import create_features_by_voltage_bins
from ml_pipeline import run_ml_pipeline

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATE_SCRIPT = os.path.join(PROJECT_DIR, "simulate_batteries.py")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# (soc_max, soc_min) pairs -- matches simulate_batteries.py's
# random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX) draw for starting SOC.
SOC_INTERVALS = [(1.0, 0.7), (0.8, 0.5), (0.6, 0.3), (0.4, 0.1)]

# All prior accuracy numbers reported in this project used the 25-step
# dataset; kept consistent here so sweep results are directly comparable.
STEP_COUNT_FOR_TRAINING = 25

RESULTS_CSV = os.path.join(DATA_DIR, "summary", "soc_sweep_results.csv")
RESULTS_COLUMNS = ["SOC_Interval", "RF_Accuracy", "XGB_Accuracy", "Top_Features"]


def _interval_label(soc_max, soc_min):
    return f"{soc_min:.1f}-{soc_max:.1f}"


def _format_top_features(feature_importances, n=3):
    ranked = sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return "; ".join(f"{name}:{importance:.4f}" for name, importance in ranked)


def _append_result_row(row):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
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

    # Per-interval run label so each interval's raw data lands in its own
    # data/soc_<label>/ tree -- sorted by run -- instead of overwriting the
    # previous interval's files, with raw/failures/plots/features sorted by
    # kind underneath it (see simulate_batteries.py's DATA_DIR/RUN_LABEL).
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

    print("[2/3] Feature engineering (voltage bins)...")
    # create_features_by_voltage_bins also writes ml_features_{n}_steps.csv
    # into data/soc_<label>/features/ as a side effect; that's ignored here
    # in favor of writing our own per-interval copy (with the SOC label in
    # its name) from the returned DataFrame directly, at a fully-qualified
    # path, so training never depends on cwd matching PROJECT_DIR or on one
    # interval's file overwriting another's.
    features_dir = os.path.join(run_dir, "features")
    datasets = create_features_by_voltage_bins(
        data_csv, step_counts=[STEP_COUNT_FOR_TRAINING], output_dir=features_dir
    )
    step_subset = datasets[STEP_COUNT_FOR_TRAINING]
    n_batteries = len(step_subset)
    print(f"  -> {n_batteries} batteries with usable {STEP_COUNT_FOR_TRAINING}-step features")

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
        "Top_Features": top_features,
    }
    _append_result_row(row)

    elapsed_min = (time.time() - t_start) / 60
    print(f"Interval {label} done in {elapsed_min:.1f} min: "
          f"RF={rf_acc:.4f} XGB={xgb_acc:.4f} ({n_batteries} batteries)")
    return row


def main():
    os.chdir(PROJECT_DIR)  # so relative paths written by the imported helpers land correctly

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
                "Top_Features": f"{type(exc).__name__}: {exc}",
            })
            continue

    total_min = (time.time() - sweep_start) / 60
    print(f"\nSweep complete in {total_min:.1f} min. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
