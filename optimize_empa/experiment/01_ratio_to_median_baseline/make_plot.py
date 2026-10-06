"""
optimize_empa, attempt 01: plots EMPA's per-SOC-specialized Random
Forest balanced accuracy vs. Initial_SOC, using the ratio-to-median
scale-invariant feature. Reads run_evaluation.py's saved
per_soc_results.json (run that first).

Usage: python3 optimize_empa/experiment/01_ratio_to_median_baseline/make_plot.py
Output: optimize_empa/experiment/01_ratio_to_median_baseline/outputs/plots/per_soc_accuracy.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "outputs", "per_soc_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "outputs", "plots", "per_soc_accuracy.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
COLOR = "#2A78D6"


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    entry = results["EMPA"]
    buckets = entry.get("buckets", {})
    rf_y = [buckets[str(s)]["balanced_accuracy"] * 100 if str(s) in buckets else None for s in SOC_START_POINTS]

    n_cells = entry.get("n_cells")
    n_cycles = entry.get("n_cycles")
    label = f"EMPA ({n_cells:,} cells, {n_cycles:,} cycles)" if n_cells and n_cycles else "EMPA"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(SOC_START_POINTS, rf_y, color=COLOR, linestyle="-", marker="o", linewidth=2, markersize=6, label=label)
    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=1.5, label="chance (50%)")

    ax.set_xlabel("Initial SOC (fraction of the cell's own capacity remaining at test time, natural cutoff)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("optimize_empa, attempt 01: per-SOC-specialized Random Forest balanced accuracy\n"
                 "(ratio-to-median scale-invariant feature -- EMPA only)")
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
