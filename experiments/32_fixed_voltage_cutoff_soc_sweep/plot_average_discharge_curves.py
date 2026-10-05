"""
Experiment 32: plots the AVERAGE real discharge curve per dataset per
chemistry, all six curves (exp07/EMPA/SNL x LFP/NMC) overlaid -- one
color per dataset (solid = LFP, dashed = NMC) -- as two subplots:
(left) Voltage vs. Charge, (right) the average dV/dQ vs. Voltage -- the
actual feature space the classifier trains on throughout this project.
Same underlying point as plot_real_discharge_curves_comparison.py (these
datasets are genuinely different, hard for one calibration to generalize
across), condensed into one clean comparison plus its feature-space
counterpart.

LEFT SUBPLOT -- AVERAGING METHOD: raw charge (Q) axis, NOT normalized to
depth-of-discharge -- matches plot_real_discharge_curves_comparison.py,
so capacity differences stay visible, not just voltage-shape
differences. Each individual trace is interpolated onto a shared per-
(dataset, chemistry) Q grid (0 to that group's own max observed
capacity); the average at each grid point is taken only over traces that
still have data there, so the average naturally stops being meaningful
(and is cut off, MIN_TRACES_FOR_AVERAGE) once too few cells remain at
deep Q.

RIGHT SUBPLOT -- AVERAGING METHOD: matches feature_engineering.py /
experiment 31's exact method, not a derivative of the smooth left-panel
curve -- dV/dQ is computed POINT-TO-POINT on each individual real trace
first (the same way the classifier's own features are built, which never
sees an averaged curve), binned into 0.1V bins
(`create_features_by_voltage_bins`, `min_chemistry_coverage=0.0` for
full-range visibility, `exclude_final_transition=True`), then the
MEDIAN per bin is plotted -- same convention as experiment 31's
dV/dQ-vs-voltage plots.

Data: Initial_SOC==1.0 (untruncated full discharge) slices of the
already-built real CSVs (no rebuild) -- same sources as
plot_real_discharge_curves_comparison.py. ALL traces used for both
averages (not subsampled -- subsampling was only for individual-trace
plot readability elsewhere).

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/plot_average_discharge_curves.py
Output: experiments/32_fixed_voltage_cutoff_soc_sweep/plots/average_discharge_curves.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

from feature_engineering import create_features_by_voltage_bins

REAL_CSVS = {
    "exp07": os.path.join(PROJECT_DIR, "data", "26_real_exp07_soc_sweep", "raw", "real_exp07_soc_sweep_raw.csv"),
    "EMPA": os.path.join(PROJECT_DIR, "data", "26_real_empa_soc_sweep_soh_range", "raw", "real_empa_soc_sweep_raw.csv"),
    "SNL": os.path.join(PROJECT_DIR, "data", "27_real_snl_soc_sweep", "raw", "real_snl_soc_sweep_raw.csv"),
}
CHEMISTRIES = ["LFP", "NMC"]
LINESTYLES = {"LFP": "-", "NMC": "--"}
COLORS = {"exp07": "#8E44AD", "EMPA": "#2A78D6", "SNL": "#1D9A6C"}

N_GRID_POINTS = 300
MIN_TRACES_FOR_AVERAGE = 5  # stop the average once fewer than this many cells still have data at that Q

TMP_DIR = os.path.join(SCRIPT_DIR, "_tmp_avg_dvdq")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", "average_discharge_curves.png")


def load_soc1_slice(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    return df[df["Initial_SOC"] == 1.0].copy()


def average_curve(chem_df):
    """Interpolates every trace onto a shared Q grid (0 to the group's
    own max observed capacity), then averages at each grid point over
    only the traces that still have data there."""
    max_q = chem_df["Capacity [A.h]"].max()
    q_grid = np.linspace(0, max_q, N_GRID_POINTS)
    voltage_matrix = []

    for vid, trace in chem_df.groupby("Variation_ID"):
        trace = trace.sort_values("Capacity [A.h]")
        cap = trace["Capacity [A.h]"].values
        volt = trace["Voltage [V]"].values
        if len(cap) < 2 or cap[-1] <= cap[0]:
            continue
        trace_max_q = cap[-1]
        v_interp = np.interp(q_grid, cap, volt, left=np.nan, right=np.nan)
        v_interp[q_grid > trace_max_q] = np.nan
        voltage_matrix.append(v_interp)

    voltage_matrix = np.array(voltage_matrix)
    n_contributing = np.sum(~np.isnan(voltage_matrix), axis=0)
    mean_voltage = np.nanmean(voltage_matrix, axis=0)

    keep = n_contributing >= MIN_TRACES_FOR_AVERAGE
    return q_grid[keep], mean_voltage[keep], len(voltage_matrix)


def bin_midpoint(col):
    parts = col.replace("dV_dQ_V_", "").split("_")
    hi, lo = float(parts[0]), float(parts[1])
    return (hi + lo) / 2.0


def average_dvdq_bins(dataset_name, df):
    """Point-to-point dV/dQ per real trace, then median per 0.1V bin --
    same method as feature_engineering.py / experiment 31, not a
    derivative of the already-averaged V-Q curve (see module docstring)."""
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_csv = os.path.join(TMP_DIR, f"{dataset_name}_real_soc1.csv")
    df.to_csv(tmp_csv, index=False)

    features_df = create_features_by_voltage_bins(
        tmp_csv, output_dir=os.path.join(TMP_DIR, dataset_name),
        exclude_final_transition=True, min_chemistry_coverage=0.0,
    )
    bin_cols = [c for c in features_df.columns if c.startswith("dV_dQ_V_")]
    medians = features_df.groupby("Chemistry")[bin_cols].median()
    return medians, bin_cols


def main():
    print("Loading real Initial_SOC=1.0 slices...")
    real_dfs = {name: load_soc1_slice(path) for name, path in REAL_CSVS.items()}

    fig, (ax_vq, ax_dvdq) = plt.subplots(1, 2, figsize=(18, 7.5))

    print("\n=== Left: average Voltage vs. Charge ===")
    for dataset_name, df in real_dfs.items():
        for chem in CHEMISTRIES:
            chem_df = df[df["Chemistry"] == chem]
            q_avg, v_avg, n_traces = average_curve(chem_df)
            ax_vq.plot(q_avg, v_avg, color=COLORS[dataset_name], linestyle=LINESTYLES[chem],
                       linewidth=2.5, label=f"{dataset_name} {chem} (n={n_traces:,})")
            print(f"  {dataset_name} {chem}: averaged over {n_traces} traces, "
                  f"grid kept from 0 to {q_avg[-1]:.2f} Ah")

    ax_vq.set_xlabel("Capacity delivered, Q [A.h]")
    ax_vq.set_ylabel("Voltage [V]")
    ax_vq.set_title("Average discharge curve\n(Voltage vs. Charge)", fontsize=13, fontweight="bold")
    ax_vq.grid(True, alpha=0.25)
    ax_vq.legend(loc="upper right", fontsize=9, framealpha=0.9)

    print("\n=== Right: average dV/dQ vs. Voltage ===")
    for dataset_name, df in real_dfs.items():
        medians, bin_cols = average_dvdq_bins(dataset_name, df)
        bin_cols_sorted = sorted(bin_cols, key=bin_midpoint, reverse=True)
        x_vals = [bin_midpoint(c) for c in bin_cols_sorted]
        for chem in CHEMISTRIES:
            if chem not in medians.index:
                continue
            y_vals = medians.loc[chem, bin_cols_sorted].values
            ax_dvdq.plot(x_vals, y_vals, color=COLORS[dataset_name], linestyle=LINESTYLES[chem],
                         marker="o", markersize=4, linewidth=2.5, label=f"{dataset_name} {chem}")
        print(f"  {dataset_name}: plotted across {len(bin_cols_sorted)} candidate bins")

    ax_dvdq.set_xlabel("Voltage [V] (bin midpoint)")
    ax_dvdq.set_ylabel("Median dV/dQ")
    ax_dvdq.set_title("Average dV/dQ\n(the classifier's actual feature space)", fontsize=13, fontweight="bold")
    ax_dvdq.invert_xaxis()
    # A handful of sparse-coverage bins at the deepest edges spike to
    # -130 to -275 (a median of only a few real points that deep, not a
    # meaningful signal -- same artifact documented in experiment 31) --
    # clipped so the real LFP-vs-NMC separation in the well-covered range
    # stays readable, not compressed into a thin band near the top.
    ax_dvdq.set_ylim(-20, 5)
    ax_dvdq.grid(True, alpha=0.25)
    ax_dvdq.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle("Real discharge behavior by dataset and chemistry (Initial_SOC = 1.0, untruncated -- solid=LFP, dashed=NMC)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
