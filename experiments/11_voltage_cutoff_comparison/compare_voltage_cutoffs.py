"""
Isolates the voltage-cutoff variable that experiment 06 left unresolved:
06 combined a randomized C-rate AND lower voltage cutoffs (LFP 1.8V->1.5V,
NMC 2.3V->1.8V) in one pass, so it's impossible to tell from 06 alone
whether the lower cutoffs alone explain its accuracy, or whether they're
masking the exact risk its own original code comment warned about (below
~2.0V "both chemistries look nearly identical").

This script holds everything else fixed -- same 0.6C continuous-discharge
protocol as experiment 03 (the main pipeline) -- and only varies the
cutoffs, at two starting-SOC conditions: root's own default (0.5-1.0,
battery starts near full) and a pinned low-SOC condition (0.1-0.4, matching
experiment 06's lowest interval) where the low-voltage region the cutoff
actually gates gets meaningfully traversed. The default-range pair turned
out uninformative on its own (see RESULTS.md) -- both cutoff variants
collapsed to the same single surviving feature bin there, because the
mutual-coverage bottleneck sits well above either cutoff at high starting
SOC. The low-SOC pair is what actually exercises the region experiment 06
flagged.

Runs root simulate_batteries.py and feature_engineering.py directly (not a
local fork): neither currently has an active caller among the experiment
sweeps (01/04 shadow the pulse-protocol archive, 03/06 use their own local
copies), so this is the first sweep to actually exercise the "main"
scripts, and the cutoffs/V_BIN_MIN needed to be made overridable there to
do it without yet another duplicated copy of the simulation code -- see
LFP_LOWER_CUTOFF/NMC_LOWER_CUTOFF in simulate_batteries.py and v_bin_min in
feature_engineering.py.

Usage: python3 experiments/11_voltage_cutoff_comparison/compare_voltage_cutoffs.py
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
sys.path.insert(0, PROJECT_DIR)  # for root feature_engineering.py and ml_pipeline.py

from feature_engineering import create_features_by_voltage_bins
from ml_pipeline import run_ml_pipeline

SIMULATE_SCRIPT = os.path.join(PROJECT_DIR, "simulate_batteries.py")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# (variant_name, lfp_cutoff, nmc_cutoff, v_bin_min) -- "original" matches
# the cutoffs used everywhere else in the repo (root, exp03); "lower"
# matches experiment 06's divergence.
CUTOFF_VARIANTS = [
    ("original", 1.8, 2.3, 1.9),
    ("lower", 1.5, 1.8, 1.5),
]

# (condition_name, soc_range_min, soc_range_max) -- None/None means leave
# root's own defaults (0.5-1.0) unset rather than duplicating them here.
SOC_CONDITIONS = [
    ("default", None, None),
    ("low_soc_0.1-0.4", 0.1, 0.4),
]

RESULTS_CSV = os.path.join(SCRIPT_DIR, "voltage_cutoff_comparison_results.csv")
RESULTS_COLUMNS = [
    "SOC_Condition", "Cutoff_Variant", "LFP_Cutoff_V", "NMC_Cutoff_V", "V_Bin_Min",
    "RF_Accuracy", "XGB_Accuracy", "Surviving_Bins", "Top_Features",
]


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


def run_one_combo(soc_condition_name, soc_min, soc_max, variant_name, lfp_cutoff, nmc_cutoff, v_bin_min):
    print(f"\n{'=' * 70}\nSOC condition '{soc_condition_name}', cutoff variant '{variant_name}' "
          f"(LFP={lfp_cutoff}V, NMC={nmc_cutoff}V, V_BIN_MIN={v_bin_min})\n{'=' * 70}")
    t_start = time.time()

    run_label = f"11_cutoff_{variant_name}" if soc_condition_name == "default" else f"11_cutoff_{variant_name}_{soc_condition_name}"
    run_dir = os.path.join(DATA_DIR, run_label)
    data_csv = os.path.join(run_dir, "raw", "advanced_synthetic_battery_data.csv")

    env = os.environ.copy()
    env["RUN_LABEL"] = run_label
    env["LFP_LOWER_CUTOFF"] = str(lfp_cutoff)
    env["NMC_LOWER_CUTOFF"] = str(nmc_cutoff)
    if soc_min is not None:
        env["SOC_RANGE_MIN"] = str(soc_min)
    if soc_max is not None:
        env["SOC_RANGE_MAX"] = str(soc_max)
    env.setdefault("MPLBACKEND", "Agg")

    print(f"[1/3] Simulating (root simulate_batteries.py, SOC condition '{soc_condition_name}', fixed 0.6C)...")
    subprocess.run([sys.executable, SIMULATE_SCRIPT], cwd=PROJECT_DIR, env=env, check=True)

    print("[2/3] Feature engineering (root feature_engineering.py, "
          f"v_bin_min={v_bin_min}, >=20% mutual-coverage filter)...")
    features_dir = os.path.join(run_dir, "features")
    step_subset = create_features_by_voltage_bins(data_csv, output_dir=features_dir, v_bin_min=v_bin_min)
    n_batteries = len(step_subset)
    surviving_bins = [c for c in step_subset.columns if c.startswith("dV_dQ_V_")]
    print(f"  -> {n_batteries} batteries, surviving bins: {surviving_bins}")

    features_csv = os.path.join(features_dir, f"ml_features_cutoff_{variant_name}.csv")
    step_subset.to_csv(features_csv, index=False)

    print("[3/3] Training (root ml_pipeline.py, Random Forest + XGBoost, artifacts not saved)...")
    results = run_ml_pipeline(synthetic_csv=features_csv, save_artifacts=False)
    matplotlib.pyplot.close("all")

    rf_acc = results["model_accuracies"].get("Random Forest")
    xgb_acc = results["model_accuracies"].get("XGBoost")
    top_features = _format_top_features(results["feature_importances"])

    row = {
        "SOC_Condition": soc_condition_name,
        "Cutoff_Variant": variant_name,
        "LFP_Cutoff_V": lfp_cutoff,
        "NMC_Cutoff_V": nmc_cutoff,
        "V_Bin_Min": v_bin_min,
        "RF_Accuracy": f"{rf_acc:.4f}" if rf_acc is not None else "",
        "XGB_Accuracy": f"{xgb_acc:.4f}" if xgb_acc is not None else "",
        "Surviving_Bins": "; ".join(surviving_bins),
        "Top_Features": top_features,
    }
    _append_result_row(row)

    elapsed_min = (time.time() - t_start) / 60
    rf_str = f"{rf_acc:.4f}" if rf_acc is not None else "n/a"
    xgb_str = f"{xgb_acc:.4f}" if xgb_acc is not None else "n/a"
    print(f"'{soc_condition_name}'/'{variant_name}' done in {elapsed_min:.1f} min: "
          f"RF={rf_str} XGB={xgb_str} ({n_batteries} batteries, {len(surviving_bins)} bins)")
    return row


def main():
    if os.path.exists(RESULTS_CSV):
        print(f"Note: {RESULTS_CSV} already exists -- new rows will be appended, not overwritten.")

    sweep_start = time.time()
    for soc_condition_name, soc_min, soc_max in SOC_CONDITIONS:
        for variant_name, lfp_cutoff, nmc_cutoff, v_bin_min in CUTOFF_VARIANTS:
            try:
                run_one_combo(soc_condition_name, soc_min, soc_max, variant_name, lfp_cutoff, nmc_cutoff, v_bin_min)
            except Exception as exc:
                print(f"!! '{soc_condition_name}'/'{variant_name}' FAILED: {type(exc).__name__}: {exc}")
                _append_result_row({
                    "SOC_Condition": soc_condition_name,
                    "Cutoff_Variant": variant_name,
                    "LFP_Cutoff_V": lfp_cutoff,
                    "NMC_Cutoff_V": nmc_cutoff,
                    "V_Bin_Min": v_bin_min,
                    "RF_Accuracy": "ERROR",
                    "XGB_Accuracy": "ERROR",
                    "Surviving_Bins": "",
                    "Top_Features": f"{type(exc).__name__}: {exc}",
                })
                continue

    total_min = (time.time() - sweep_start) / 60
    print(f"\nComparison complete in {total_min:.1f} min. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
