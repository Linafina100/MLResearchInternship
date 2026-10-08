import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "sweep_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "soc_sweep_accuracy_empa.png")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

# EMPA styling configuration
EMPA_COLOR = "#2A78D6"


def main():
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    if "EMPA" not in results:
        print("EMPA dataset not found in sweep_results.json.")
        return

    buckets = results["EMPA"].get("buckets", {})
    rf_y = [
        buckets[str(s)]["Random Forest"]["balanced_accuracy"] * 100
        if str(s) in buckets else None
        for s in SOC_START_POINTS
    ]
    xgb_y = [
        buckets[str(s)]["XGBoost"]["balanced_accuracy"] * 100
        if str(s) in buckets else None
        for s in SOC_START_POINTS
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(SOC_START_POINTS, rf_y, color=EMPA_COLOR, linestyle="-", marker="o",
            linewidth=2, markersize=6, label="EMPA - Random Forest")
    ax.plot(SOC_START_POINTS, xgb_y, color=EMPA_COLOR, linestyle="--", marker="s",
            linewidth=2, markersize=6, alpha=0.8, label="EMPA - XGBoost")

    n_first = buckets.get("1.0", {}).get("Random Forest", {}).get("n")
    if n_first and rf_y[0] is not None:
        ax.annotate(f"n={n_first}", xy=(1.0, rf_y[0]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color=EMPA_COLOR)

    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=1.5, label="Chance (50%)")

    ax.set_xlabel("Initial SOC (fraction of cell's capacity remaining)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("EMPA Sim-to-Real Accuracy vs. Initial SOC\n"
                 "(Continuous-discharge-truncated synthetic training, all voltage bins)")
    ax.set_xlim(1.05, 0.0)  # Descending: full charge (left) -> near-empty (right)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()