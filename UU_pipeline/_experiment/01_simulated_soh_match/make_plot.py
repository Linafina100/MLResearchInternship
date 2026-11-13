"""
Plots the pooled-model SOC-sweep accuracy curve (RF + XGBoost) 
from the UU pipeline's outputs/sweep_results.json.

Usage: python3 UU_pipeline/make_plot_soc_sweep.py
Output: UU_pipeline/outputs/plots/soc_sweep_accuracy_UU.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "outputs", "sweep_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "outputs", "plots", "soc_sweep_accuracy_UU.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

# Distinct, professional colors for the UU plot curves
RF_COLOR = "#1f77b4"   # Blue
XGB_COLOR = "#ff7f0e"  # Orange


def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"Results JSON not found at {RESULTS_JSON}. Run evaluation first.")
        return

    with open(RESULTS_JSON) as f:
        results = json.load(f)

    # Check for "UU" key or fallback to the first available key in results
    key = "UU" if "UU" in results else (list(results.keys())[0] if results else None)
    if not key:
        print("No dataset results found in sweep_results.json.")
        return

    dataset_results = results[key]
    buckets = dataset_results.get("buckets", {})
    meta = dataset_results.get("meta", {})

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

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(SOC_START_POINTS, rf_y, color=RF_COLOR, linestyle="-", marker="o",
            linewidth=2, markersize=6, label="UU - Random Forest")
    ax.plot(SOC_START_POINTS, xgb_y, color=XGB_COLOR, linestyle="--", marker="s",
            linewidth=2, markersize=6, alpha=0.8, label="UU - XGBoost")

    if meta.get("n_cells") and meta.get("n_cycles"):
        ax.text(0.98, 0.02, f"{meta['n_cells']} cells, {meta['n_cycles']} cycles",
                transform=ax.transAxes, fontsize=9, color="#555555",
                ha="right", va="bottom")

    ax.set_xlabel("Initial SOC (fraction of cell's capacity remaining)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("01_simulated_soh_match: UU Sim-to-Real Accuracy vs. Initial SOC")
    ax.set_xlim(1.05, 0.0)  # Descending: full charge (left) -> near-empty (right)
    ax.set_ylim(0, 105)     # Show the full 0-100% picture
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()