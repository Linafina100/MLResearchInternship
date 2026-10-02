"""
Experiment 32: plots Random Forest balanced accuracy vs. Initial_SOC for
exp07, EMPA, and SNL, all rebuilt against the fixed 2.5V universal cutoff
voltage. Reads evaluate_soc_sweep_fixed_cutoff.py's saved
sweep_results.json (run that first). Legend entries include each
dataset's total unique physical cells and total discharge cycles (pooled
across all 12 Initial_SOC points), as requested for this experiment.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/make_soc_sweep_plot.py
Output: experiments/32_fixed_voltage_cutoff_soc_sweep/plots/soc_sweep_accuracy.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "sweep_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "soc_sweep_accuracy.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

# Same per-dataset colors used throughout this project's SOC-sweep plots
# (experiments/22_29_sim_to_real_validation/27_snl_soc_sweep/make_soc_sweep_plot.py),
# kept consistent so a reader who has seen those plots recognizes these
# datasets by the same hue.
STYLE = {
    "exp07": {"color": "#8E44AD"},
    "EMPA": {"color": "#2A78D6"},
    "SNL": {"color": "#1D9A6C"},
}


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 6))

    for dataset_name, style in STYLE.items():
        if dataset_name not in results:
            continue
        entry = results[dataset_name]
        buckets = entry.get("buckets", {})
        rf_y = [buckets[str(s)]["Random Forest"]["balanced_accuracy"] * 100 if str(s) in buckets else None for s in SOC_START_POINTS]

        n_cells = entry.get("n_cells")
        n_cycles = entry.get("n_cycles")
        label = f"{dataset_name} ({n_cells:,} cells, {n_cycles:,} cycles)" if n_cells and n_cycles else dataset_name

        ax.plot(SOC_START_POINTS, rf_y, color=style["color"], linestyle="-", marker="o",
                linewidth=2, markersize=6, label=label)

    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=1.5, label="chance (50%)")

    ax.set_xlabel("Initial SOC (fraction of the cell's own capacity remaining at test time, 0% = 2.5V)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Sim-to-real balanced accuracy vs. starting charge level, Random Forest\n"
                 "(fixed 2.5V universal cutoff voltage -- exp07 / EMPA / SNL)")
    ax.set_xlim(1.05, 0.0)  # descending: full charge (left) -> near-empty (right)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
