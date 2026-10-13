"""
optimize_empa_low_soc: campaign-level plot. Reads every attempt's
outputs/sweep_results.json and plots XGBoost balanced accuracy vs.
Initial_SOC on one chart, showing the progression across the campaign
-- each attempt isolated exactly one variable (see each attempt's own
RESULTS.md and the campaign index for what changed and why).

Random Forest is omitted: the campaign index's cross-attempt comparison
found XGBoost wins or ties at every SOC point across every attempt
tested, so it's not part of the "best so far" story this plot tells.

Attempt 02 is also omitted: confirmed its XGBoost balanced accuracy is
within 0.02pp of attempt 03's at every single SOC point (the factor
tuning in attempt 03 only moved Random Forest, not XGBoost -- see
attempt 03's RESULTS.md) -- its line would render perfectly hidden
under attempt 03's, adding legend clutter with no visible information.

Usage: python3 optimize_empa_low_soc/make_plot.py
Output: optimize_empa_low_soc/outputs/plots/campaign_progress.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(SCRIPT_DIR, "outputs", "plots", "campaign_progress.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

ATTEMPTS = [
    ("01_baseline_carryover", "01: baseline (carried over)", "#9AA4AC", 1.5, "--"),
    ("03_tuned_negative_electrode_factor", "02-03: negative-electrode balance (tuned)", "#5B9BD5", 1.8, "-"),
    ("04_ratio_to_median_features", "04: + ratio-to-median", "#ED8B3F", 1.8, "-"),
    ("05_stable_median_reference", "05: + stable median reference (current best)", "#2A9D5C", 2.6, "-"),
]


def load_xgb_balanced(attempt_folder):
    path = os.path.join(SCRIPT_DIR, "experiment", attempt_folder, "outputs", "sweep_results.json")
    with open(path) as f:
        buckets = json.load(f)["EMPA"]["buckets"]
    return [buckets[str(s)]["XGBoost"]["balanced_accuracy"] * 100 if str(s) in buckets else None for s in SOC_START_POINTS]


def main():
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for folder, label, color, lw, ls in ATTEMPTS:
        y = load_xgb_balanced(folder)
        ax.plot(SOC_START_POINTS, y, color=color, linestyle=ls, linewidth=lw,
                 marker="o", markersize=5, label=label,
                 zorder=5 if folder.startswith("05") else 3)

    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=1.2, label="chance (50%)", zorder=1)

    ax.set_xlabel("Initial SOC (fraction of the cell's own capacity remaining at test time)")
    ax.set_ylabel("XGBoost balanced accuracy (%)")
    ax.set_title("optimize_empa_low_soc: campaign progress\n(EMPA real-data sim-to-real sweep, XGBoost)")
    ax.set_xlim(1.05, 0.0)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
