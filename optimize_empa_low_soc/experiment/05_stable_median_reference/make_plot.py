"""
optimize_empa_low_soc, attempt 05: stakeholder-facing summary plot.
Reads this attempt's own outputs/sweep_results.json (the campaign's
current best pipeline) and plots overall accuracy alongside each
chemistry's own detection rate (recall), against how much charge a
battery has left when it's tested -- the real-world question this
answers: "if a battery arrives at an unknown, possibly low charge
level, can we still tell LFP from NMC?"

Deliberately free of implementation jargon (no model names, no
feature-engineering terms) -- see this folder's RESULTS.md for the
full technical derivation this plot summarizes.

Usage: python3 optimize_empa_low_soc/experiment/05_stable_median_reference/make_plot.py
Output: optimize_empa_low_soc/experiment/05_stable_median_reference/outputs/plots/stakeholder_summary.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "outputs", "sweep_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "outputs", "plots", "stakeholder_summary.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

COLOR_OVERALL = "#1F4E79"
COLOR_LFP = "#2F9E44"
COLOR_NMC = "#E07B1A"
COLOR_CHANCE = "#ADB5BD"


def main():
    with open(RESULTS_JSON) as f:
        buckets = json.load(f)["EMPA"]["buckets"]

    soc_pct = [s * 100 for s in SOC_START_POINTS]
    overall, lfp, nmc, n_total = [], [], [], []
    for s in SOC_START_POINTS:
        b = buckets[str(s)]["XGBoost"]
        overall.append(b["balanced_accuracy"] * 100)
        lfp.append(b["LFP_recall"] * 100)
        nmc.append(b["NMC_recall"] * 100)
        n_total.append(buckets[str(s)]["n"])

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.axhline(50, color=COLOR_CHANCE, linestyle=":", linewidth=1.3, zorder=1, label="Random guessing")

    ax.plot(soc_pct, nmc, color=COLOR_NMC, linewidth=2.2, marker="o", markersize=6,
             label="NMC batteries correctly identified", zorder=3)
    ax.plot(soc_pct, lfp, color=COLOR_LFP, linewidth=2.2, marker="o", markersize=6,
             label="LFP batteries correctly identified", zorder=3)
    ax.plot(soc_pct, overall, color=COLOR_OVERALL, linewidth=3.4, marker="o", markersize=7,
             label="Overall accuracy", zorder=4)

    ax.annotate("LFP identified correctly\nat every charge level tested",
                xy=(30, 100), xytext=(55, 88),
                fontsize=9.5, color=COLOR_LFP, ha="center",
                arrowprops=dict(arrowstyle="->", color=COLOR_LFP, lw=1.2))

    ax.annotate("Still >90% reliable even at\njust 10% charge remaining",
                xy=(10, 95.2), xytext=(35, 65),
                fontsize=9.5, color=COLOR_OVERALL, ha="center",
                arrowprops=dict(arrowstyle="->", color=COLOR_OVERALL, lw=1.2))

    ax.set_xlabel("Remaining charge when tested (% of the battery's own capacity)", fontsize=11)
    ax.set_ylabel("Correct classification rate (%)", fontsize=11)

    ax.set_xlim(102, 3)
    ax.set_ylim(45, 103)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(loc="lower left", fontsize=10, framealpha=0.95, borderpad=0.8)

    fig.suptitle("Telling LFP and NMC batteries apart, at any charge level",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.935,
             "Validated against real discharge data from 199 physical EMPA battery cells",
             ha="center", fontsize=10, color="#495057")

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
