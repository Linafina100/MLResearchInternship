"""
Experiment 31: purely visual/diagnostic comparison -- real SNL and EMPA
discharge curves plotted directly against the synthetic calibration data
(the same continuous-discharge-truncated baseline used throughout
experiments 26-29), in both raw voltage and dV/dQ space. This makes the
calibration mismatch that experiments 27-29 already described numerically
(SNL's real LFP dV/dQ magnitude vs. the exp21 calibration) actually
visible, for both the dataset that works (EMPA) and the one that doesn't
(SNL), side by side.

No new data, no resimulation -- reuses the existing untruncated
(Initial_SOC==1.0) slices of already-built CSVs:
- Synthetic: data/26_soh_range_continuous_discharge_truncated_v1/
- Real EMPA: data/26_real_empa_soc_sweep_soh_range/
- Real SNL: data/27_real_snl_soc_sweep/

Two figures, each a 2x2 grid (rows: SNL / EMPA, columns: LFP / NMC):
1. Voltage vs. depth-of-discharge (each curve normalized to its own
   fraction of total delivered capacity, so differently-sized cells
   overlay comparably). Real traces subsampled for readability; synthetic
   traces shown in full (249 per chemistry, already a manageable count).
2. dV/dQ vs. voltage -- the feature space the classifier actually uses.
   Reuses feature_engineering.create_features_by_voltage_bins() unchanged,
   but with min_chemistry_coverage=0.0 (no leakage-style filtering -- this
   is diagnostic, not training, so showing the full voltage range
   including sparse-coverage bins is the point). Plots the MEDIAN dV/dQ
   per bin as one clean line per (real/synthetic, chemistry) group.

Usage: python3 experiments/31_snl_empa_curve_comparison/make_curve_comparison_plots.py
Output: experiments/31_snl_empa_curve_comparison/plots/voltage_vs_dod.png
        experiments/31_snl_empa_curve_comparison/plots/dvdq_vs_voltage.png
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

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "26_soh_range_continuous_discharge_truncated_v1", "raw", "advanced_synthetic_battery_data.csv")
REAL_CSVS = {
    "SNL": os.path.join(PROJECT_DIR, "data", "27_real_snl_soc_sweep", "raw", "real_snl_soc_sweep_raw.csv"),
    "EMPA": os.path.join(PROJECT_DIR, "data", "26_real_empa_soc_sweep_soh_range", "raw", "real_empa_soc_sweep_raw.csv"),
}
DATASETS = ["SNL", "EMPA"]
CHEMISTRIES = ["LFP", "NMC"]
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

COLOR_REAL = "#D9552B"
COLOR_SYNTH = "#2A6FDB"
MAX_REAL_TRACES_PLOTTED = 150
RNG = np.random.default_rng(42)

VOLTAGE_PLOT_PNG = os.path.join(SCRIPT_DIR, "plots", "voltage_vs_dod.png")
DVDQ_PLOT_PNG = os.path.join(SCRIPT_DIR, "plots", "dvdq_vs_voltage.png")


def load_soc1_slice(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    return df[df["Initial_SOC"] == 1.0].copy()


def subsample_variation_ids(df, max_n):
    ids = df["Variation_ID"].unique()
    if len(ids) <= max_n:
        return ids
    return RNG.choice(ids, size=max_n, replace=False)


def plot_voltage_vs_dod():
    print("=== Figure 1: voltage vs. depth-of-discharge ===")
    synth_df = load_soc1_slice(SYNTHETIC_RAW_CSV)
    real_dfs = {name: load_soc1_slice(path) for name, path in REAL_CSVS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)

    for row, dataset_name in enumerate(DATASETS):
        real_df = real_dfs[dataset_name]
        for col, chem in enumerate(CHEMISTRIES):
            ax = axes[row, col]

            synth_chem = synth_df[synth_df["Chemistry"] == chem]
            synth_ids = subsample_variation_ids(synth_chem, MAX_REAL_TRACES_PLOTTED)
            for i, vid in enumerate(synth_ids):
                trace = synth_chem[synth_chem["Variation_ID"] == vid].sort_values("Time [s]")
                cap = trace["Capacity [A.h]"].values
                if cap.max() <= 0:
                    continue
                dod = cap / cap.max()
                ax.plot(dod, trace["Voltage [V]"], color=COLOR_SYNTH, alpha=0.12, linewidth=0.8,
                        label="Synthetic (calibration)" if i == 0 else None, zorder=2)

            real_chem = real_df[real_df["Chemistry"] == chem]
            real_ids = subsample_variation_ids(real_chem, MAX_REAL_TRACES_PLOTTED)
            for i, vid in enumerate(real_ids):
                trace = real_chem[real_chem["Variation_ID"] == vid].sort_values("Time [s]")
                cap = trace["Capacity [A.h]"].values
                if cap.max() <= 0:
                    continue
                dod = cap / cap.max()
                ax.plot(dod, trace["Voltage [V]"], color=COLOR_REAL, alpha=0.12, linewidth=0.8,
                        label=f"Real {dataset_name}" if i == 0 else None, zorder=3)

            n_real = real_chem["Variation_ID"].nunique()
            n_synth = synth_chem["Variation_ID"].nunique()
            ax.set_title(f"{dataset_name} -- {chem}  (real n={n_real:,}, synthetic n={n_synth})", fontsize=11)
            ax.set_xlabel("Depth of discharge (fraction of own capacity)")
            ax.set_ylabel("Voltage [V]")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
            print(f"  {dataset_name} {chem}: plotted {min(n_real, MAX_REAL_TRACES_PLOTTED)} real / "
                  f"{min(n_synth, MAX_REAL_TRACES_PLOTTED)} synthetic traces")

    fig.suptitle("Real vs. synthetic discharge curves (Initial_SOC = 1.0, untruncated)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(VOLTAGE_PLOT_PNG), exist_ok=True)
    plt.savefig(VOLTAGE_PLOT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {VOLTAGE_PLOT_PNG}")


def compute_bin_medians(dataset_name, real_csv):
    combined_csv = os.path.join(SCRIPT_DIR, f"_combined_soc1_{dataset_name}.csv")
    features_dir = os.path.join(SCRIPT_DIR, "_features_soc1", dataset_name)

    synth_df = load_soc1_slice(SYNTHETIC_RAW_CSV)
    synth_df["DataKind"] = "synthetic"
    real_df = load_soc1_slice(real_csv)
    real_df["DataKind"] = "real"
    combined = pd.concat([synth_df, real_df], ignore_index=True)
    combined.to_csv(combined_csv, index=False)  # always overwritten

    features_df = create_features_by_voltage_bins(
        combined_csv, output_dir=features_dir, exclude_final_transition=True, min_chemistry_coverage=0.0,
    )

    raw_df = pd.read_csv(combined_csv, low_memory=False)
    raw_df["Battery_ID"] = raw_df.groupby(GROUPBY_COLS).ngroup()
    id_cols = raw_df.drop_duplicates("Battery_ID").set_index("Battery_ID")["DataKind"]
    features_df["DataKind"] = features_df["Battery_ID"].map(id_cols)

    bin_cols = [c for c in features_df.columns if c.startswith("dV_dQ_V_")]
    medians = features_df.groupby(["DataKind", "Chemistry"])[bin_cols].median()
    return medians, bin_cols


def bin_midpoint(col):
    parts = col.replace("dV_dQ_V_", "").split("_")
    hi, lo = float(parts[0]), float(parts[1])
    return (hi + lo) / 2.0


def plot_dvdq_vs_voltage():
    print("\n=== Figure 2: dV/dQ vs. voltage ===")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=False)

    for row, dataset_name in enumerate(DATASETS):
        medians, bin_cols = compute_bin_medians(dataset_name, REAL_CSVS[dataset_name])
        bin_cols_sorted = sorted(bin_cols, key=bin_midpoint, reverse=True)
        x_vals = [bin_midpoint(c) for c in bin_cols_sorted]

        for col, chem in enumerate(CHEMISTRIES):
            ax = axes[row, col]
            for kind, color, label in [("real", COLOR_REAL, f"Real {dataset_name}"),
                                        ("synthetic", COLOR_SYNTH, "Synthetic (calibration)")]:
                if (kind, chem) not in medians.index:
                    continue
                y_vals = medians.loc[(kind, chem), bin_cols_sorted].values
                ax.plot(x_vals, y_vals, color=color, marker="o", markersize=4, linewidth=2, label=label)

            ax.set_title(f"{dataset_name} -- {chem}", fontsize=11)
            ax.set_xlabel("Voltage [V] (bin midpoint)")
            ax.set_ylabel("Median dV/dQ")
            ax.invert_xaxis()
            ax.grid(True, alpha=0.25)
            ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
            print(f"  {dataset_name} {chem}: plotted across {len(bin_cols_sorted)} candidate bins")

    fig.suptitle("Real vs. synthetic dV/dQ by voltage (Initial_SOC = 1.0, all candidate bins, no coverage filter)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(DVDQ_PLOT_PNG), exist_ok=True)
    plt.savefig(DVDQ_PLOT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {DVDQ_PLOT_PNG}")


if __name__ == "__main__":
    plot_voltage_vs_dod()
    plot_dvdq_vs_voltage()
