"""
Experiment 28: 2x2 grid, one subplot per real dataset (exp07, EMPA, CALCE,
SNL), comparing the POOLED Random Forest (experiments 26/27 -- one model
trained on all 12 Initial_SOC truncation points pooled) against the
PER-SOC-SPECIALIZED Random Forest (this experiment -- one model per
Initial_SOC point) on the same balanced-accuracy-vs-Initial_SOC axes.

Usage: python3 experiments/28_per_soc_point_models/make_comparison_plot.py
Output: experiments/28_per_soc_point_models/plots/pooled_vs_specialized.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
POOLED_JSON = os.path.join(PROJECT_DIR, "experiments", "27_snl_soc_sweep", "sweep_results.json")
SPECIALIZED_JSON = os.path.join(SCRIPT_DIR, "per_soc_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "pooled_vs_specialized.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
DATASETS = ["exp07", "EMPA", "CALCE", "SNL"]
COLOR_POOLED = "#9AA4AC"
COLOR_SPECIALIZED = "#2A78D6"


def main():
    with open(POOLED_JSON) as f:
        pooled = json.load(f)
    with open(SPECIALIZED_JSON) as f:
        specialized = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for ax, dataset_name in zip(axes.flat, DATASETS):
        pooled_buckets = pooled.get(dataset_name, {}).get("buckets", {})
        pooled_y = [pooled_buckets[str(s)]["Random Forest"]["balanced_accuracy"] * 100
                    if str(s) in pooled_buckets else None for s in SOC_START_POINTS]

        spec_buckets = specialized.get(dataset_name, {})
        spec_y = [spec_buckets[str(s)]["balanced_accuracy"] * 100
                  if str(s) in spec_buckets else None for s in SOC_START_POINTS]

        ax.plot(SOC_START_POINTS, pooled_y, color=COLOR_POOLED, linestyle="--", marker="s",
                linewidth=2, markersize=5, label="Pooled (one model, all SOC)")
        ax.plot(SOC_START_POINTS, spec_y, color=COLOR_SPECIALIZED, linestyle="-", marker="o",
                linewidth=2, markersize=5, label="Per-SOC-specialized")
        ax.axhline(50, color="#C9CFD4", linestyle=":", linewidth=1.2)

        ax.set_title(dataset_name, fontsize=12, fontweight="bold")
        ax.set_xlim(1.05, 0.0)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Initial SOC")
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced accuracy (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Pooled vs. per-SOC-specialized Random Forest, balanced accuracy vs. Initial SOC",
                 fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
