"""
Experiment 32: SMOKE TEST ONLY (not the full pipeline) for a new,
scale-invariant feature engineering approach.

WHY: every experiment in this project so far trains on ABSOLUTE dV/dQ
values. Those values scale with the battery's physical size (Ah) --
confirmed visually in plot_average_discharge_curves.py's dV/dQ panel,
where exp07 (large cells), EMPA (coin cells), and SNL (small cylindrical
cells) show very different magnitudes even within one chemistry. At the
Stena Recycling production line, a battery arrives at an unknown SOC/SOH
and only a PARTIAL discharge window is observed -- there is no way to
know that cell's true total capacity to normalize against. So a feature
space built on absolute dV/dQ is implicitly asking the classifier to
learn "which dataset is this from" (a size signature) as much as "which
chemistry is this" -- not a one-off path check in the previous commit,
but a design flaw to fix on a branch of its own.

THE IDEA: replace absolute per-bin dV/dQ with a SCALE-INVARIANT,
WINDOW-RELATIVE version computed using only the observed window's own
bins (no external normalization, no total-capacity knowledge needed) --
two candidate transforms, applied ROW-WISE (per sample/battery, using
only that row's own non-NaN bins):
  - Z-SCORE: (bin_value - row_mean) / row_std
  - RATIO-TO-MEDIAN: bin_value / row_median

THIS SCRIPT: a smoke test only, at a PARTIAL window (Initial_SOC=0.5 --
the realistic production case, not the easy full-discharge case) across
all three real datasets (exp07/EMPA/SNL), both chemistries. Reuses the
EXISTING, UNMODIFIED root feature_engineering.create_features_by_voltage_bins()
to extract raw per-bin dV/dQ first (same point-to-point method every
other experiment uses), then applies the two transforms on TOP of that
output -- no changes to the shared root utility, contained entirely to
this experiment. No rebuild needed: reuses the already-built real CSVs
from data/26_real_exp07_soc_sweep/, data/26_real_empa_soc_sweep_soh_range/,
data/27_real_snl_soc_sweep/.

VERIFICATION: for each transform (raw / z-score / ratio-to-median),
plots the median value per bin per (dataset, chemistry) group -- same
style as plot_average_discharge_curves.py's dV/dQ panel -- so it's
directly visible whether the transform pulls the three datasets' curves
closer together (successful alignment) while still keeping LFP and NMC
visibly distinct within each dataset (chemistry signal preserved, not
washed out).

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/smoke_test_scale_invariant_features.py
Output: experiments/32_fixed_voltage_cutoff_soc_sweep/plots/smoke_test_scale_invariant_features.png
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

SMOKE_TEST_SOC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5  # partial window -- the realistic production case, not SOC=1.0

TMP_DIR = os.path.join(SCRIPT_DIR, "_tmp_scale_invariant_smoke")
OUT_PNG = os.path.join(SCRIPT_DIR, "plots", f"smoke_test_scale_invariant_features_v2_ratio_soc_{SMOKE_TEST_SOC}.png")


def bin_midpoint(col):
    parts = col.replace("dV_dQ_V_", "").split("_")
    hi, lo = float(parts[0]), float(parts[1])
    return (hi + lo) / 2.0


def extract_bins_at_soc(dataset_name, csv_path, soc):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["Initial_SOC"] == soc].copy()

    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_csv = os.path.join(TMP_DIR, f"{dataset_name}_soc_{soc}.csv")
    df.to_csv(tmp_csv, index=False)

    # FIX 1: standard >=20% mutual-coverage filter (the project default),
    # not 0.0 -- the first smoke test's min_chemistry_coverage=0.0 let
    # extremely sparse bins through, which is what made EMPA NMC swing
    # from +25 to -150 in the raw panel.
    features_df = create_features_by_voltage_bins(
        tmp_csv, output_dir=os.path.join(TMP_DIR, dataset_name),
        exclude_final_transition=True,
    )
    bin_cols = sorted([c for c in features_df.columns if c.startswith("dV_dQ_V_")], key=bin_midpoint, reverse=True)
    return features_df, bin_cols


MIN_BINS_PER_ROW = 3  # FIX 2: guard against zero/near-zero-variance rows
                       # (too few observed bins) producing NaN/inf ratios


def apply_ratio_to_median(features_df, bin_cols):
    """Row-wise (per sample), using only that sample's own non-NaN
    observed bins -- no cross-sample or cross-dataset information. Rows
    with fewer than MIN_BINS_PER_ROW valid bins are dropped entirely
    (not just left as NaN) -- a ratio computed from 1-2 points on a
    heavily truncated curve isn't a meaningful "shape," just noise."""
    X = features_df[bin_cols].values.astype(float)
    n_valid = np.sum(~np.isnan(X), axis=1)
    keep_mask = n_valid >= MIN_BINS_PER_ROW

    n_dropped = (~keep_mask).sum()
    if n_dropped:
        print(f"    dropped {n_dropped}/{len(X)} rows with < {MIN_BINS_PER_ROW} valid bins")

    X_kept = X[keep_mask]
    row_median = np.nanmedian(X_kept, axis=1, keepdims=True)
    ratio = X_kept / row_median

    ratio_df = pd.DataFrame(ratio, columns=bin_cols, index=features_df.index[keep_mask])
    ratio_df["Chemistry"] = features_df["Chemistry"].values[keep_mask]
    return ratio_df


def plot_panel(ax, per_dataset_medians, bin_cols, title, ylabel):
    x_vals = [bin_midpoint(c) for c in bin_cols]
    for dataset_name, medians in per_dataset_medians.items():
        for chem in CHEMISTRIES:
            if chem not in medians.index:
                continue
            y_vals = medians.loc[chem, bin_cols].values
            ax.plot(x_vals, y_vals, color=COLORS[dataset_name], linestyle=LINESTYLES[chem],
                     marker="o", markersize=3, linewidth=2, label=f"{dataset_name} {chem}")
    ax.set_xlabel("Voltage [V] (bin midpoint)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def main():
    print(f"Smoke test v2 (ratio-to-median only, >=20% coverage filter reinstated, "
          f"min_bins_per_row={MIN_BINS_PER_ROW}): Initial_SOC={SMOKE_TEST_SOC} (partial window), "
          f"all three real datasets, both chemistries.\n")

    raw_medians, ratio_medians = {}, {}
    all_bin_cols = set()

    for dataset_name, csv_path in REAL_CSVS.items():
        features_df, bin_cols = extract_bins_at_soc(dataset_name, csv_path, SMOKE_TEST_SOC)
        all_bin_cols.update(bin_cols)
        print(f"{dataset_name}: {len(features_df)} batteries, {len(bin_cols)} candidate bins (>=20% coverage)")
        ratio_df = apply_ratio_to_median(features_df, bin_cols)

        raw_medians[dataset_name] = features_df.groupby("Chemistry")[bin_cols].median()
        ratio_medians[dataset_name] = ratio_df.groupby("Chemistry")[bin_cols].median()

        for chem in CHEMISTRIES:
            if chem in raw_medians[dataset_name].index:
                n_raw = (features_df["Chemistry"] == chem).sum()
                n_ratio = (ratio_df["Chemistry"] == chem).sum() if chem in ratio_df["Chemistry"].values else 0
                print(f"  {chem}: n_raw={n_raw}, n_after_min_bins_guard={n_ratio}, raw median range "
                      f"[{raw_medians[dataset_name].loc[chem, bin_cols].min():.2f}, "
                      f"{raw_medians[dataset_name].loc[chem, bin_cols].max():.2f}]")

    common_bin_cols = sorted(all_bin_cols, key=bin_midpoint, reverse=True)

    def reindexed(medians_dict):
        return {k: v.reindex(columns=common_bin_cols) for k, v in medians_dict.items()}

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plot_panel(axes[0], reindexed(raw_medians), common_bin_cols, "Raw dV/dQ (current method)", "Median dV/dQ")
    plot_panel(axes[1], reindexed(ratio_medians), common_bin_cols, "Ratio to row median (fixed)", "Median x / row_median")

    fig.suptitle(f"Smoke test v2: ratio-to-median, >=20% coverage + min_bins guard "
                 f"(Initial_SOC={SMOKE_TEST_SOC}, partial window)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
