"""
Experiment 26: plots balanced accuracy vs. Initial_SOC for all three real
datasets (experiment 07, EMPA, CALCE), RF and XGBoost each as a separate
line -- reads evaluate_soc_sweep_three_datasets.py's saved
sweep_results.json (run that first).

Usage: python3 experiments/26_soc_sweep_three_datasets/make_soc_sweep_plot.py
Output: experiments/26_soc_sweep_three_datasets/plots/soc_sweep_accuracy.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "sweep_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "soc_sweep_accuracy.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
EXTRAPOLATION_POINTS = {0.4, 0.3}

# dataset -> (color, RF linestyle/marker, XGB linestyle/marker)
STYLE = {
    "exp07": {"color": "#8E44AD"},
    "EMPA": {"color": "#2A78D6"},
    "CALCE": {"color": "#EB6834"},
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
        xgb_y = [buckets[str(s)]["XGBoost"]["balanced_accuracy"] * 100 if str(s) in buckets else None for s in SOC_START_POINTS]

        ax.plot(SOC_START_POINTS, rf_y, color=style["color"], linestyle="-", marker="o",
                linewidth=2, markersize=6, label=f"{dataset_name} - Random Forest")
        ax.plot(SOC_START_POINTS, xgb_y, color=style["color"], linestyle="--", marker="s",
                linewidth=2, markersize=6, alpha=0.7, label=f"{dataset_name} - XGBoost")

        n_first = buckets.get("1.0", {}).get("Random Forest", {}).get("n")
        if n_first:
            ax.annotate(f"n={n_first}", xy=(1.0, rf_y[0]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color=style["color"])

    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=1.5, label="chance (50%)")
    ax.axvspan(0.25, 0.45, color="#FAB219", alpha=0.08, zorder=0)
    ax.text(0.35, ax.get_ylim()[0] if False else 3, "beyond synthetic\ntraining range",
            fontsize=8, color="#B8860B", ha="center", va="bottom")

    ax.set_xlabel("Initial SOC (fraction of the cell's own capacity remaining at test time)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Sim-to-real balanced accuracy vs. starting charge level\n"
                 "(trained once per dataset on synthetic SOH 0.8-1.0 data, all voltage bins)")
    ax.set_xlim(1.05, 0.25)  # descending: full charge (left) -> near-empty (right)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
