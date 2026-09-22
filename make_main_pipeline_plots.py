"""
Main-pipeline diagnostic plots, saved to data/simulating_problems/: does
the fully-fixed main simulate_batteries.py (1.5V cutoff, SOH 0.8-1.0,
C-rate 0.1-0.2, temp 15-35C, SOH-scaling both concentrations, t_interp,
NMC diffusivity/10, LFP OCP rate=-3) close the real-vs-synthetic gap that
data/simulating_problems/'s earlier plots documented?

Uses whatever synthetic run RUN_LABEL points at (run
simulate_batteries.py first) and the same real LFP/NMC parsers every
sim-to-real experiment uses.

Produces (date-stamped so a rerun doesn't overwrite a prior one):
1. main_pipeline_sim_vs_real_<date>.png -- raw discharge curves + dV/dQ
   profiles, real vs. synthetic overlaid (same style as experiment 16's
   make_illustrative_plots.py plot 3).
2. main_pipeline_coverage_and_magnitude_<date>.png -- target-zone
   (2.5-3.0V) per-bin coverage and mean dV/dQ magnitude, real vs.
   synthetic, both chemistries.

Usage: RUN_LABEL=<synthetic run> python3 make_main_pipeline_plots.py
       (defaults to RUN_LABEL=main_pipeline_<today's date>)
"""
import os
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP07_DIR = os.path.join(PROJECT_DIR, "experiments", "07_real_lfp_nmc_test")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, EXP07_DIR)

from feature_engineering import create_features_by_voltage_bins
from parse_real_lfp import parse_lfp_discharge_files
from parse_real_nmc import parse_nmc_files

TODAY = date.today().isoformat()
RUN_LABEL = os.environ.get("RUN_LABEL", f"main_pipeline_{TODAY}")
SYNTH_RAW_CSV = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "raw", "synthetic_battery_data.csv")

OUT_DIR = os.path.join(PROJECT_DIR, "data", "simulating_problems")
WORK_DIR = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "main_pipeline_plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

RAW_CSV = os.path.join(WORK_DIR, "sim_and_real_raw.csv")
FEATURES_DIR = os.path.join(WORK_DIR, "features")

GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
CHEM_COLOR = {"LFP": "#1f77b4", "NMC": "#ff7f0e"}
KIND_STYLE = {"real": {"linestyle": "-", "marker": "o"}, "synthetic": {"linestyle": "--", "marker": "s"}}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True,
    "grid.alpha": 0.3, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "font.size": 9,
})


def build_combined_raw_csv():
    if not os.path.exists(SYNTH_RAW_CSV):
        raise FileNotFoundError(f"'{SYNTH_RAW_CSV}' missing -- run simulate_batteries.py first "
                                 f"(RUN_LABEL={RUN_LABEL}).")
    sim_df = pd.read_csv(SYNTH_RAW_CSV)
    sim_df["DataKind"] = "synthetic"

    lfp_df = parse_lfp_discharge_files()
    nmc_df = parse_nmc_files()
    real_df = pd.concat([lfp_df, nmc_df], ignore_index=True)
    real_df["DataKind"] = "real"

    combined = pd.concat([sim_df, real_df], ignore_index=True)
    combined.to_csv(RAW_CSV, index=False)
    print(f"Combined raw data: {(combined['DataKind'] == 'synthetic').sum()} synthetic rows, "
          f"{(combined['DataKind'] == 'real').sum()} real rows -> '{RAW_CSV}'")
    return combined


def extract_features_with_kind():
    features_df = create_features_by_voltage_bins(RAW_CSV, output_dir=FEATURES_DIR)
    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df["Battery_ID"] = raw_df.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_df.drop_duplicates("Battery_ID").set_index("Battery_ID")["DataKind"]
    features_df["DataKind"] = features_df["Battery_ID"].map(battery_to_kind)
    return features_df


def plot_sim_vs_real(raw, features_df):
    """Same style as experiment 16's make_illustrative_plots.py plot 3."""
    bin_cols = [c for c in features_df.columns if c.startswith("dV_dQ_V_")]
    v_axis = [(float(c.split("_")[-2]) + float(c.split("_")[-1])) / 2.0 for c in bin_cols]
    sorted_pairs = sorted(zip(v_axis, bin_cols), reverse=True)
    v_axis_sorted = [p[0] for p in sorted_pairs]
    sorted_cols = [p[1] for p in sorted_pairs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    rng = np.random.RandomState(3)
    legend_added = set()
    for (chem, kind), group in raw.groupby(["Chemistry", "DataKind"]):
        battery_ids = group["Battery_ID"].unique()
        n_sample = min(20, len(battery_ids))
        sampled = rng.choice(battery_ids, size=n_sample, replace=False)
        color, style = CHEM_COLOR[chem], KIND_STYLE[kind]
        for bid in sampled:
            g = group[group["Battery_ID"] == bid].sort_values("Time [s]")
            key = (chem, kind)
            ax1.plot(g["Capacity [A.h]"], g["Voltage [V]"], color=color, linestyle=style["linestyle"],
                      alpha=0.18, linewidth=1.0,
                      label=f"{kind.capitalize()} {chem}" if key not in legend_added else None)
            legend_added.add(key)

    ax1.set_title("Simulated vs. Real: Discharge Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Discharged Capacity [Ah]", fontsize=11)
    ax1.set_ylabel("Terminal Voltage [V]", fontsize=11)
    ax1.axhspan(TARGET_ZONE_MIN, TARGET_ZONE_MAX, color="orange", alpha=0.1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    leg1 = ax1.legend(loc="upper right", frameon=True, fontsize=9)
    for line in leg1.get_lines():
        line.set_alpha(1.0)
        line.set_linewidth(2.0)

    for (chem, kind), group in features_df.groupby(["Chemistry", "DataKind"]):
        color, style = CHEM_COLOR[chem], KIND_STYLE[kind]
        for _, row in group.iterrows():
            y_vals = row[sorted_cols].astype(float).values
            ax2.plot(v_axis_sorted, y_vals, color=color, linestyle=style["linestyle"], alpha=0.08, linewidth=0.8)
        mean_vals = group[sorted_cols].mean(skipna=True).values
        ax2.plot(v_axis_sorted, mean_vals, color=color, linestyle=style["linestyle"], marker=style["marker"],
                  linewidth=2.5, markersize=5, label=f"{kind.capitalize()} {chem} (mean)")

    ax2.set_title("Simulated vs. Real: dV/dQ Feature Profiles", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cell Terminal Voltage [V] (bin center)", fontsize=11)
    ax2.set_ylabel("dV/dQ [V/Ah]", fontsize=11)
    ax2.invert_xaxis()
    ax2.axvspan(TARGET_ZONE_MIN, TARGET_ZONE_MAX, color="orange", alpha=0.1, label="Target zone (2.5-3.0V)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=8)

    fig.suptitle(f"Main pipeline ({TODAY}) vs. real data: raw discharge shape and dV/dQ profile", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, f"main_pipeline_sim_vs_real_{TODAY}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def target_zone_bin_cols(all_bin_cols):
    kept = []
    for col in all_bin_cols:
        parts = col.split("_")
        bin_high, bin_low = float(parts[-2]), float(parts[-1])
        if bin_low >= TARGET_ZONE_MIN and bin_high <= TARGET_ZONE_MAX:
            kept.append(col)
    return sorted(kept, key=lambda c: float(c.split("_")[-1]))


def plot_coverage_and_magnitude(features_df):
    bin_cols = target_zone_bin_cols([c for c in features_df.columns if c.startswith("dV_dQ_V_")])
    bin_labels = [f"{c.split('_')[-1]}-{c.split('_')[-2]}" for c in bin_cols]
    x = np.arange(len(bin_labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, chem in enumerate(("LFP", "NMC")):
        real_sub = features_df[(features_df.DataKind == "real") & (features_df.Chemistry == chem)]
        synth_sub = features_df[(features_df.DataKind == "synthetic") & (features_df.Chemistry == chem)]

        ax_cov = axes[row, 0]
        real_cov = real_sub[bin_cols].notna().mean().values * 100
        synth_cov = synth_sub[bin_cols].notna().mean().values * 100
        ax_cov.bar(x - width / 2, real_cov, width, label="Real", color="tab:blue")
        ax_cov.bar(x + width / 2, synth_cov, width, label="Synthetic (main pipeline)", color="tab:red")
        ax_cov.set_xticks(x)
        ax_cov.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
        ax_cov.set_ylabel("% of batteries with a sample in this bin")
        ax_cov.set_title(f"{chem}: target-zone coverage")
        ax_cov.legend()

        ax_mag = axes[row, 1]
        real_mean = real_sub[bin_cols].mean().values
        synth_mean = synth_sub[bin_cols].mean().values
        ax_mag.bar(x - width / 2, real_mean, width, label="Real", color="tab:blue")
        ax_mag.bar(x + width / 2, synth_mean, width, label="Synthetic (main pipeline)", color="tab:red")
        ax_mag.set_xticks(x)
        ax_mag.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
        ax_mag.set_ylabel("Mean dV/dQ [V/Ah]")
        ax_mag.set_title(f"{chem}: target-zone dV/dQ magnitude")
        ax_mag.legend()

    fig.suptitle(f"Main pipeline ({TODAY}): target-zone (2.5-3.0V) coverage and magnitude, real vs. synthetic", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, f"main_pipeline_coverage_and_magnitude_{TODAY}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    build_combined_raw_csv()
    features_df = extract_features_with_kind()
    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df["Battery_ID"] = raw_df.groupby(GROUPBY_COLS).ngroup()

    print(features_df.groupby(["DataKind", "Chemistry"]).size().rename("n_batteries"))
    plot_sim_vs_real(raw_df, features_df)
    plot_coverage_and_magnitude(features_df)


if __name__ == "__main__":
    main()
