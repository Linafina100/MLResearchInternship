"""
Plots the pooled-model SOC-sweep accuracy curve (RF + XGBoost) from this
experiment's outputs/sweep_results.json, alongside experiment 01's original
(buggy-NMC-parsing) curve for comparison.

Usage: python3 UU_pipeline/_experiment/02_nmc_contiguous_segment_fix/make_plot.py
Output: UU_pipeline/_experiment/02_nmc_contiguous_segment_fix/outputs/plots/soc_sweep_accuracy_UU.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "outputs", "sweep_results.json")
PRIOR_RESULTS_JSON = os.path.join(SCRIPT_DIR, "..", "01_simulated_soh_match", "outputs", "sweep_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "outputs", "plots", "soc_sweep_accuracy_UU.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

RF_COLOR = "#1f77b4"      # Blue
XGB_COLOR = "#ff7f0e"     # Orange
PRIOR_COLOR = "#999999"   # Grey, for the buggy exp01 baseline


def load_curves(path):
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        results = json.load(f)
    key = "UU" if "UU" in results else (list(results.keys())[0] if results else None)
    if not key:
        return None, None
    buckets = results[key].get("buckets", {})
    rf_y = [
        buckets[str(s)]["Random Forest"]["balanced_accuracy"] * 100
        if str(s) in buckets and "Random Forest" in buckets[str(s)] else None
        for s in SOC_START_POINTS
    ]
    xgb_y = [
        buckets[str(s)]["XGBoost"]["balanced_accuracy"] * 100
        if str(s) in buckets and "XGBoost" in buckets[str(s)] else None
        for s in SOC_START_POINTS
    ]
    return rf_y, xgb_y


def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"Results JSON not found at {RESULTS_JSON}. Run run_evaluation.py first.")
        return

    rf_y, xgb_y = load_curves(RESULTS_JSON)
    prior_rf_y, _ = load_curves(PRIOR_RESULTS_JSON)

    fig, ax = plt.subplots(figsize=(8, 5))

    if prior_rf_y:
        ax.plot(SOC_START_POINTS, prior_rf_y, color=PRIOR_COLOR, linestyle=":", marker="x",
                linewidth=1.5, markersize=5, alpha=0.8, label="exp01 (buggy NMC parsing) - RF")

    ax.plot(SOC_START_POINTS, rf_y, color=RF_COLOR, linestyle="-", marker="o",
            linewidth=2, markersize=6, label="exp02 (fixed) - Random Forest")
    ax.plot(SOC_START_POINTS, xgb_y, color=XGB_COLOR, linestyle="--", marker="s",
            linewidth=2, markersize=6, alpha=0.8, label="exp02 (fixed) - XGBoost")

    ax.set_xlabel("Initial SOC (fraction of cell's capacity remaining)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("02_nmc_contiguous_segment_fix: UU Sim-to-Real Accuracy vs. Initial SOC")
    ax.set_xlim(1.05, 0.0)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
