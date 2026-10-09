"""
Genererar en presentationstålig visualisering enbart för EMPA-datasetet
baserat på mallen från experiment 29.
Vad som ändrats:
* Rensat bort exp07: Alla referenser och datahämtningar för det andra datasetet har plockats bort.
* Fokuserat på EMPA: Skriptet läser nu enbart in resultaten för EMPA och ritar upp kurvan med dess specifika färg (#2A78D6) och etiketter.
*Anpassade axlar: SOC_POINTS har anpassats till [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3] (du kan enkelt lägga till fler i listan om du kör en mer utökad sweep ner till 0.05!).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Anpassa sökvägen till där din per-SOC JSON för EMPA ligger
PER_SOC_JSON = os.path.join(PROJECT_DIR, "optimize_empa_2", "per_soc_results.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "poc_empa_only.png")

# EMPA-specifika SOC-punkter (anpassa listan om dina punkter skiljer sig)
SOC_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

SERIES = {
    "EMPA": {"display_label": "EMPA public research dataset", "color": "#2A78D6", "source": "per_soc"},
}


def get_series_data(name, source, per_soc):
    """Hämtar (balanced_accuracy_pct lista, n) för EMPA."""
    buckets = per_soc[name]
    y = [buckets[str(s)]["balanced_accuracy"] * 100 for s in SOC_POINTS]
    n = buckets[str(SOC_POINTS[0])]["n"]
    return y, n


def main():
    with open(PER_SOC_JSON) as f:
        per_soc = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, style in SERIES.items():
        y, n = get_series_data(name, style["source"], per_soc)
        label = f"{style['display_label']} (n={n:,})"
        ax.plot(SOC_POINTS, y, color=style["color"], linestyle="-", marker="o",
                linewidth=3, markersize=8, label=label, zorder=3)

        # Annotera startvärdet direkt på kurvan
        ax.annotate(f"{y[0]:.0f}%", xy=(SOC_POINTS[0], y[0]), xytext=(-8, 14),
                    textcoords="offset points", fontsize=13, fontweight="bold",
                    color=style["color"], ha="right")

    ax.axhline(50, color="#9AA4AC", linestyle=":", linewidth=2, zorder=1)
    ax.annotate("Random guessing (50%)", xy=(0.35, 50), xytext=(0, 6),
                textcoords="offset points", fontsize=11, color="#6b7680", ha="left")

    ax.set_xlabel("Battery's remaining charge when tested (fraction of a full charge)", fontsize=13, labelpad=12)
    ax.set_ylabel("Correct chemistry classification rate (%)", fontsize=13, labelpad=12)
    ax.set_title(
        "EMPA Dataset Validation on Physics-Based Simulated Model",
        fontsize=16, fontweight="bold", pad=18,
    )
    ax.text(
        0.5, 1.005,
        "Model trained only on physics-based computer simulations, never shown EMPA data during training",
        transform=ax.transAxes, ha="center", fontsize=10.5, color="#5a6570", style="italic",
    )

    ax.set_xlim(max(SOC_POINTS) + 0.05, min(SOC_POINTS) - 0.05)
    ax.set_ylim(0, 105)
    ax.set_xticks(SOC_POINTS)
    ax.set_xticklabels([f"{int(s*100)}%" for s in SOC_POINTS], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend = ax.legend(loc="lower left", fontsize=12, framealpha=0.95, title="Validation dataset",
                        title_fontsize=12)
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=170)
    print(f"Saved EMPA-only plot to {OUT_PNG}")


if __name__ == "__main__":
    main()