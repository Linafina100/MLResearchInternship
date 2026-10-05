"""
Experiment 32: plots the three real datasets' (exp07, EMPA, SNL) own
discharge curves -- raw Voltage [V] vs. Capacity [A.h] (charge
delivered, NOT normalized to depth-of-discharge or rescaled to a common
capacity) -- directly against each other, one panel per chemistry. The
point: make visually obvious what RESULTS.md's dV/dQ analysis already
measured numerically -- these three datasets' real cells are genuinely
different (different capacities, different voltage-vs-charge shapes),
which is why one shared synthetic calibration can't generalize to all
three at once (see "A real cost: this is not a free lunch for exp07 or
all of SNL" in RESULTS.md).

No synthetic data here deliberately -- this is real vs. real (unlike
experiment 31's real vs. synthetic comparison) -- and no normalization,
per explicit request: raw V vs. Q lets the very different absolute
capacities (exp07 LFP up to 5.7Ah, EMPA rescaled flat to 2.0Ah, SNL LFP
only ~1.1Ah) show up on the plot too, not just the voltage shape.

Data: Initial_SOC==1.0 (untruncated full discharge) slices of the
already-built real CSVs (no rebuild) --
- exp07: data/26_real_exp07_soc_sweep/
- EMPA: data/26_real_empa_soc_sweep_soh_range/
- SNL: data/27_real_snl_soc_sweep/
Real traces subsampled to MAX_TRACES_PLOTTED per panel (EMPA alone has
tens of thousands at this SOC point), low alpha, same convention as
experiment 31's make_curve_comparison_plots.py.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/plot_real_discharge_curves_comparison.py
Output: experiments/32_fixed_voltage_cutoff_soc_sweep/plots/real_discharge_curves_comparison.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

REAL_CSVS = {
    "exp07": os.path.join(PROJECT_DIR, "data", "26_real_exp07_soc_sweep", "raw", "real_exp07_soc_sweep_raw.csv"),
    "EMPA": os.path.join(PROJECT_DIR, "data", "26_real_empa_soc_sweep_soh_range", "raw", "real_empa_soc_sweep_raw.csv"),
    "SNL": os.path.join(PROJECT_DIR, "data", "27_real_snl_soc_sweep", "raw", "real_snl_soc_sweep_raw.csv"),
}
CHEMISTRIES = ["LFP", "NMC"]
MAX_TRACES_PLOTTED = 150
RNG = np.random.default_rng(42)

COLORS = {"exp07": "#8E44AD", "EMPA": "#2A78D6", "SNL": "#1D9A6C"}

OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "real_discharge_curves_comparison.png")


def load_soc1_slice(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    return df[df["Initial_SOC"] == 1.0].copy()


def subsample_variation_ids(df, max_n):
    ids = df["Variation_ID"].unique()
    if len(ids) <= max_n:
        return ids
    return RNG.choice(ids, size=max_n, replace=False)


def main():
    print("Loading real Initial_SOC=1.0 slices...")
    real_dfs = {name: load_soc1_slice(path) for name, path in REAL_CSVS.items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

    for col, chem in enumerate(CHEMISTRIES):
        ax = axes[col]
        for dataset_name, df in real_dfs.items():
            chem_df = df[df["Chemistry"] == chem]
            ids = subsample_variation_ids(chem_df, MAX_TRACES_PLOTTED)
            n_total = chem_df["Variation_ID"].nunique()
            for i, vid in enumerate(ids):
                trace = chem_df[chem_df["Variation_ID"] == vid].sort_values("Time [s]")
                ax.plot(trace["Capacity [A.h]"], trace["Voltage [V]"], color=COLORS[dataset_name],
                         alpha=0.15, linewidth=0.8,
                         label=f"{dataset_name} (n={n_total:,})" if i == 0 else None)
            print(f"  {chem} {dataset_name}: plotted {min(n_total, MAX_TRACES_PLOTTED)} of {n_total} real traces")

        ax.set_title(f"{chem}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Capacity delivered, Q [A.h]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    axes[0].set_ylabel("Voltage [V]")

    fig.suptitle("Real discharge curves by dataset (Voltage vs. Charge, Initial_SOC = 1.0, untruncated)\n"
                 "Different capacities, different shapes -- one calibration can't fit all three",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
