import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "sweep_results_4.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "soc_sweep_accuracy_empa_4.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

# Distinct styling for all 4 models
MODEL_CONFIGS = {
    "Random Forest": {"color": "#1f77b4", "linestyle": "-",  "marker": "o", "label": "Random Forest"},
    "XGBoost":       {"color": "#ff7f0e", "linestyle": "--", "marker": "s", "label": "XGBoost"},
    "MLP":           {"color": "#2ca02c", "linestyle": "-.", "marker": "^", "label": "MLP (Neural Net)"},
    "RBF Network":   {"color": "#d62728", "linestyle": ":",  "marker": "D", "label": "RBF Network (SVC)"},
}


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    if "EMPA" not in results:
        print("EMPA dataset not found in sweep_results.json.")
        return

    buckets = results["EMPA"].get("buckets", {})

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model_name, cfg in MODEL_CONFIGS.items():
        y_vals = [
            buckets[str(s)][model_name]["balanced_accuracy"] * 100
            if str(s) in buckets and model_name in buckets[str(s)] else None
            for s in SOC_START_POINTS
        ]

        ax.plot(
            SOC_START_POINTS, y_vals,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            marker=cfg["marker"],
            linewidth=2,
            markersize=6,
            alpha=0.9,
            label=cfg["label"]
        )

    # Annotate sample count at SOC=1.0 if available
    first_bucket = buckets.get("1.0", {})
    if first_bucket:
        any_model = next(iter(first_bucket.values()), {})
        n_first = any_model.get("n")
        first_acc = any_model.get("balanced_accuracy", 0) * 100
        if n_first:
            ax.annotate(f"n={n_first}", xy=(1.0, first_acc), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color="#333333")

    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=1.5, label="Chance (50%)")

    ax.set_xlabel("Initial SOC (fraction of cell's capacity remaining)", fontsize=11)
    ax.set_ylabel("Balanced accuracy (%)", fontsize=11)
    ax.set_title("EMPA Sim-to-Real Accuracy vs. Initial SOC Across Models\n"
                 "(Trained on Synthetic PyBaMM, Tested on Real EMPA Traces)", fontsize=12, fontweight="bold")
    
    ax.set_xlim(1.05, 0.0)  # Descending: full charge (left) -> near-empty (right)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()