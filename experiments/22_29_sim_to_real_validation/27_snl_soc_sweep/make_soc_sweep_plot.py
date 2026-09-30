"""
Experiment 27: plots Random Forest balanced accuracy vs. Initial_SOC for
all FOUR real datasets (experiment 07, EMPA, CALCE, SNL) -- RF only
(experiment 26 established XGBoost isn't fit for this task; not repeated
here). Reads evaluate_soc_sweep_four_datasets.py's saved
sweep_results.json (run that first).

Usage: python3 experiments/22_29_sim_to_real_validation/27_snl_soc_sweep/make_soc_sweep_plot.py
Output: experiments/22_29_sim_to_real_validation/27_snl_soc_sweep/plots/soc_sweep_accuracy.png
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
# No extrapolation zone: the continuous-discharge-truncated synthetic
# training set (data/26_soh_range_continuous_discharge_truncated_v1/)
# covers this entire grid in-distribution for all four datasets.

# dataset -> color
STYLE = {
    "exp07": {"color": "#8E44AD"},
    "EMPA": {"color": "#2A78D6"},
    "CALCE": {"color": "#EB6834"},
    "SNL": {"color": "#1D9A6C"},
}


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 6))

    for dataset_name, style in STYLE.items():
        if dataset_name not in results:
            continue
        buckets = results[dataset_name].get("buckets", {})
        rf_y = [buckets[str(s)]["Random Forest"]["balanced_accuracy"] * 100 if str(s) in buckets else None for s in SOC_START_POINTS]

        ax.plot(SOC_START_POINTS, rf_y, color=style["color"], linestyle="-", marker="o",
                linewidth=2, markersize=6, label=f"{dataset_name} - Random Forest")

        n_first = buckets.get("1.0", {}).get("Random Forest", {}).get("n")
        if n_first:
            ax.annotate(f"n={n_first}", xy=(1.0, rf_y[0]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color=style["color"])

    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=1.5, label="chance (50%)")

    ax.set_xlabel("Initial SOC (fraction of the cell's own capacity remaining at test time)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Sim-to-real balanced accuracy vs. starting charge level, Random Forest\n"
                 "(continuous-discharge-truncated synthetic training, all voltage bins, 4 real datasets)")
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
