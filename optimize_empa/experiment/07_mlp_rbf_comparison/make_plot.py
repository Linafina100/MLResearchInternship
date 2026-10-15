"""
optimize_empa, experiment 07: plots the pooled-model SOC-sweep accuracy
curve for all four models (Random Forest, XGBoost, MLP, RBF SVM) from
this experiment's own outputs/sweep_results.json. Adapted from
experiment 06's make_plot.py -- four distinct colors instead of one
color/varied-linestyle, since all four models are expected to land
close together (~97-99%) and need to stay visually distinguishable.

Usage: python3 optimize_empa/experiment/07_mlp_rbf_comparison/make_plot.py
Output: optimize_empa/experiment/07_mlp_rbf_comparison/outputs/plots/soc_sweep_accuracy_empa.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "outputs", "sweep_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "outputs", "plots", "soc_sweep_accuracy_empa.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

MODEL_STYLE = {
    "Random Forest": {"color": "#2A78D6", "marker": "o"},
    "XGBoost":        {"color": "#E67E22", "marker": "s"},
    "MLP":            {"color": "#27AE60", "marker": "^"},
    "RBF SVM":        {"color": "#8E44AD", "marker": "D"},
}


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    if "EMPA" not in results:
        print("EMPA dataset not found in sweep_results.json.")
        return

    buckets = results["EMPA"].get("buckets", {})
    meta = results["EMPA"].get("meta", {})

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, style in MODEL_STYLE.items():
        y = [
            buckets[str(s)][model_name]["balanced_accuracy"] * 100
            if str(s) in buckets and model_name in buckets[str(s)] else None
            for s in SOC_START_POINTS
        ]
        ax.plot(SOC_START_POINTS, y, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=6, label=model_name)

    if meta.get("n_cells") and meta.get("n_cycles"):
        ax.text(0.98, 0.02, f"{meta['n_cells']} cells, {meta['n_cycles']} cycles",
                transform=ax.transAxes, fontsize=9, color="#555555",
                ha="right", va="bottom")

    ax.set_xlabel("Initial SOC (fraction of cell's capacity remaining)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Sim-to-Real Accuracy vs. Initial SOC")
    ax.set_xlim(1.05, 0.0)  # Descending: full charge (left) -> near-empty (right)
    ax.set_ylim(0, 105)  # not clipped to a narrow high-accuracy band -- show the full picture either way
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
