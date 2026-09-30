"""
Experiment 29: presentation-ready proof-of-concept plot for stakeholders --
classification accuracy vs. how much charge is left in the battery, for
two independent real-world datasets (experiment 07, EMPA). No
resimulation needed here, this just re-presents already-computed numbers
with stakeholder-appropriate labeling:
- exp07: experiments 26/27's pooled Random Forest (one model trained
  across all 12 Initial_SOC points) -- unchanged, already excellent.
- EMPA: experiment 28's PER-SOC-SPECIALIZED Random Forest (one model per
  Initial_SOC point) instead of the pooled one, per explicit request --
  see experiments/22_29_sim_to_real_validation/28_per_soc_point_models/RESULTS.md for how this differs
  from pooled (near-perfect through Initial_SOC=0.5, then a discrete drop
  once the surviving bin composition changes, rather than pooled's
  smoother decline).

Legend "n" = total real CYCLE count in that dataset's test set (not
physical battery/cell count).

SNL is deliberately NOT included in this plot -- see RESULTS.md for why
(a genuine, investigated calibration limitation, not hidden but not
presented alongside a "capability" plot either).

Usage: python3 experiments/22_29_sim_to_real_validation/29_snl_lfp_ocp_recalibration/make_poc_comparison_plot.py
Output: experiments/22_29_sim_to_real_validation/29_snl_lfp_ocp_recalibration/plots/poc_comparison_exp07_empa.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
POOLED_JSON = os.path.join(PROJECT_DIR, "experiments", "22_29_sim_to_real_validation", "27_snl_soc_sweep", "sweep_results.json")
PER_SOC_JSON = os.path.join(PROJECT_DIR, "experiments", "22_29_sim_to_real_validation", "28_per_soc_point_models", "per_soc_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "poc_comparison_exp07_empa.png")

SOC_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

SERIES = {
    "exp07": {"display_label": "In-house lab cells", "color": "#8E44AD", "source": "pooled"},
    "EMPA": {"display_label": "EMPA public research dataset", "color": "#2A78D6", "source": "per_soc"},
}


def get_series_data(name, source, pooled, per_soc):
    """Returns (balanced_accuracy_pct list, n) for the requested source."""
    if source == "pooled":
        buckets = pooled[name]["buckets"]
        y = [buckets[str(s)]["Random Forest"]["balanced_accuracy"] * 100 for s in SOC_POINTS]
        n = buckets[str(SOC_POINTS[0])]["Random Forest"]["n"]
    else:
        buckets = per_soc[name]
        y = [buckets[str(s)]["balanced_accuracy"] * 100 for s in SOC_POINTS]
        n = buckets[str(SOC_POINTS[0])]["n"]
    return y, n


def main():
    with open(POOLED_JSON) as f:
        pooled = json.load(f)
    with open(PER_SOC_JSON) as f:
        per_soc = json.load(f)

    fig, ax = plt.subplots(figsize=(11, 7))

    for name, style in SERIES.items():
        y, n = get_series_data(name, style["source"], pooled, per_soc)
        label = f"{style['display_label']} (n={n:,})"
        ax.plot(SOC_POINTS, y, color=style["color"], linestyle="-", marker="o",
                linewidth=3, markersize=8, label=label, zorder=3)

        # Annotate the headline (full-charge) number directly on the curve.
        ax.annotate(f"{y[0]:.0f}%", xy=(SOC_POINTS[0], y[0]), xytext=(-8, 14),
                    textcoords="offset points", fontsize=13, fontweight="bold",
                    color=style["color"], ha="right")

    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=2, zorder=1)
    ax.annotate("Random guessing (50%)", xy=(0.06, 50), xytext=(0, 6),
                textcoords="offset points", fontsize=11, color="#6b7680", ha="left")

    ax.set_xlabel("Battery's remaining charge when tested (fraction of a full charge)", fontsize=13, labelpad=12)
    ax.set_ylabel("Correct chemistry classification rate (%)", fontsize=13, labelpad=12)
    ax.set_title(
        "Testing two independent datasets on model trained on simulated data",
        fontsize=16, fontweight="bold", pad=18,
    )
    ax.text(
        0.5, 1.005,
        "Model trained only on physics-based computer simulations, never shown either real dataset during training",
        transform=ax.transAxes, ha="center", fontsize=10.5, color="#5a6570", style="italic",
    )

    ax.set_xlim(1.05, 0.0)
    ax.set_ylim(0, 105)
    ax.set_xticks(SOC_POINTS)
    ax.set_xticklabels([f"{int(s*100)}%" for s in SOC_POINTS], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend = ax.legend(loc="lower left", fontsize=12, framealpha=0.95, title="Real-world validation dataset",
                        title_fontsize=12)
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=170)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
