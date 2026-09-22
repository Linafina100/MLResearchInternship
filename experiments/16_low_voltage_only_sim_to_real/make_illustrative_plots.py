"""
Two diagnostic plots for experiment 16, saved to data/simulating_problems/:

1. artifact_cutoff_comparison.png -- the problem the cutoff hack SOLVED.
   Raw discharge curves (Voltage vs. Capacity) comparing the original
   2.3V NMC cutoff (data/11_cutoff_original/, where the solver's abrupt
   event-triggered termination produces a visible final jump near/inside
   the 2.5-3.0V real-world target zone) against the lowered 1.5V cutoff
   this experiment uses (data/16_low_voltage_v1.5/, where the curve instead
   passes smoothly through the target zone, artifact pushed well below
   it).

2. sampling_density_coverage_gap.png -- the SEPARATE problem the cutoff
   hack did NOT solve. Two panels: (a) per-bin coverage in 2.5-3.0V,
   real NMC (~100%) vs. synthetic NMC (0-0.9%) -- PyBaMM's adaptive
   solver leaves synthetic NMC almost empty there regardless of cutoff
   (computed directly from raw traces, real + synthetic); (b) per-bin
   mean dV/dQ magnitude, real vs. synthetic LFP, showing the systematic
   2-6x mismatch (reuses the already-extracted, already-validated
   features/ml_features.csv from evaluate_low_voltage_sim_to_real.py, so
   these numbers match RESULTS.md's Phase 2 table exactly). Run that
   script first if features/ml_features.csv doesn't exist yet.

3. sim_vs_real_comparison.png -- side-by-side raw discharge curves
   (Voltage vs. Capacity) and dV/dQ profiles (vs. voltage bin), real and
   synthetic overlaid on the same axes. Replicates experiment 10's
   plotting style exactly (see
   experiments/10_constant_current_three/simulate_batteries_const_three.py
   and feature_engineering_const_three.py's plot_dvdq_profiles): chemistry
   colors #1f77b4 (LFP) / #ff7f0e (NMC), thin low-alpha individual
   trajectories with a bold mean overlay, dashed grid, bold titles.

Usage: python3 experiments/16_low_voltage_only_sim_to_real/evaluate_low_voltage_sim_to_real.py  (once, if not already run)
       python3 experiments/16_low_voltage_only_sim_to_real/make_illustrative_plots.py
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
EXP07_DIR = os.path.join(PROJECT_DIR, "experiments", "07_real_lfp_nmc_test")
sys.path.insert(0, EXP07_DIR)  # for parse_real_lfp.py / parse_real_nmc.py

from parse_real_nmc import parse_nmc_files

OUT_DIR = os.path.join(PROJECT_DIR, "data", "simulating_problems")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True,
    "grid.alpha": 0.3, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "font.size": 9,
})

TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0

ORIGINAL_CUTOFF_CSV = os.path.join(PROJECT_DIR, "data", "11_cutoff_original", "raw", "advanced_synthetic_battery_data.csv")
LOW_CUTOFF_CSV = os.path.join(PROJECT_DIR, "data", "16_low_voltage_v1.5", "raw", "advanced_synthetic_battery_data.csv")

GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']


def pick_batteries(df, chem, prefer_param_set=None, n=3, seed=1):
    sub = df[df["Chemistry"] == chem].copy()
    sub["Battery_ID"] = sub.groupby(GROUPBY_COLS).ngroup()
    ids = sub["Battery_ID"].unique()
    if prefer_param_set and "Base_Parameter_Set" in sub.columns:
        preferred = sub[sub["Base_Parameter_Set"] == prefer_param_set]["Battery_ID"].unique()
        if len(preferred) >= n:
            ids = preferred
    rng = np.random.RandomState(seed)
    chosen = rng.choice(ids, size=min(n, len(ids)), replace=False)
    return sub, chosen


# ---------------------------------------------------------------------------
# Plot 1: raw discharge curves, old vs. new cutoff -- the artifact itself
# ---------------------------------------------------------------------------

def plot_1_artifact_cutoff_comparison():
    df_old, df_new = pd.read_csv(ORIGINAL_CUTOFF_CSV), pd.read_csv(LOW_CUTOFF_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, df, cutoff, title in [
        (axes[0], df_old, 2.3, "Original cutoff (2.3V):\nsolver terminates with an abrupt final jump near the target zone"),
        (axes[1], df_new, 1.5, "Lowered cutoff (1.5V):\ncurve passes smoothly through the target zone"),
    ]:
        sub, chosen = pick_batteries(df, "NMC", prefer_param_set="Mohtat2020", n=3, seed=2)
        for i, bid in enumerate(chosen):
            g = sub[sub["Battery_ID"] == bid].sort_values("Time [s]")
            v, q = g["Voltage [V]"].values, g["Capacity [A.h]"].values
            ax.plot(q, v, marker='o', markersize=3, linewidth=1, color=f"C{i}",
                    alpha=0.85, label=f"NMC battery {i+1}" if i < 3 else None)

            # Highlight the single FINAL transition -- the pathological,
            # event-triggered oversized step -- distinctly from the rest
            # of the (normal) discharge knee.
            dv, dq = v[-1] - v[-2], q[-1] - q[-2]
            if dq > 1e-5:
                dvdq = dv / dq
                ax.plot(q[-2:], v[-2:], color="red", linewidth=2.5, zorder=5,
                         label="Final transition (artifact)" if (ax is axes[0] and i == 0) else None)
                ax.annotate(f"dV/dQ={dvdq:.0f}", xy=(q[-1], v[-1]), xytext=(8, -4 if i else 10),
                            textcoords="offset points", fontsize=7, color="red")

        ax.axhspan(TARGET_ZONE_MIN, TARGET_ZONE_MAX, color="orange", alpha=0.15,
                   label="Real-world target zone (2.5-3.0V)" if ax is axes[0] else None)
        ax.axhline(cutoff, color="black", linestyle="--", linewidth=1,
                   label="Simulated cutoff" if ax is axes[0] else None)
        ax.set_xlabel("Discharge capacity [A.h]")
        ax.set_title(title)

    axes[0].set_ylabel("Voltage [V]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Problem 1 (solved): the solver-termination artifact, before and after lowering the simulated cutoff -- NMC batteries",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    out_path = os.path.join(OUT_DIR, "artifact_cutoff_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: sampling-density / coverage gap -- the problem NOT solved
# ---------------------------------------------------------------------------

def per_battery_transitions(groups):
    """Yield (v_curr array, dV/dQ array) per group, discharge-direction transitions only."""
    for _, g in groups:
        g = g.sort_values("Time [s]")
        v, q = g["Voltage [V]"].values, g["Capacity [A.h]"].values
        dV, dQ = np.diff(v), np.diff(q)
        valid = dQ > 1e-5
        yield v[1:][valid], (dV[valid] / dQ[valid])


def coverage_by_bin(groups, bin_edges):
    n_total, counts = 0, np.zeros(len(bin_edges) - 1)
    for v_curr, _ in per_battery_transitions(groups):
        n_total += 1
        idx = np.digitize(v_curr, bin_edges) - 1
        for b in set(int(i) for i in idx if 0 <= i < len(counts)):
            counts[b] += 1
    return (counts / n_total * 100) if n_total else counts


def shade_zone(ax, bin_labels, bin_edges):
    rounded_edges = [round(float(e), 1) for e in bin_edges[:-1]]
    start = rounded_edges.index(round(TARGET_ZONE_MIN, 1))
    end = rounded_edges.index(round(TARGET_ZONE_MAX - 0.1, 1))
    ax.axvspan(start - 0.5, end + 0.5, color="orange", alpha=0.15, label="Target zone (2.5-3.0V)")


def plot_2_sampling_density_coverage_gap():
    """Sources: (a) NMC coverage computed directly from raw traces (real
    parser + synthetic raw CSV) since coverage isn't itself a feature-
    engineering output; (b) LFP magnitude reuses the already-extracted,
    already-validated `features/ml_features.csv` from
    evaluate_low_voltage_sim_to_real.py (root feature_engineering.py,
    exclude_final_transition=True) so these numbers match RESULTS.md's
    Phase 2 table exactly, rather than a fresh, less-robust recomputation.
    """
    real_nmc = parse_nmc_files()
    synth_all = pd.read_csv(LOW_CUTOFF_CSV)
    synth_nmc = synth_all[synth_all["Chemistry"] == "NMC"].copy()

    features_csv = os.path.join(SCRIPT_DIR, "features", "ml_features.csv")
    raw_combined_csv = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")
    if not (os.path.exists(features_csv) and os.path.exists(raw_combined_csv)):
        raise FileNotFoundError(
            "Missing features/ml_features.csv or sim_and_real_raw.csv -- "
            "run evaluate_low_voltage_sim_to_real.py first."
        )
    features_df = pd.read_csv(features_csv)
    raw_combined = pd.read_csv(raw_combined_csv, low_memory=False)
    raw_combined["Battery_ID"] = raw_combined.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_combined.drop_duplicates("Battery_ID").set_index("Battery_ID")["DataKind"]
    features_df["DataKind"] = features_df["Battery_ID"].map(battery_to_kind)

    feature_bin_cols = [c for c in features_df.columns if c.startswith("dV_dQ_V_")]
    # Column name is "dV_dQ_V_{high}_{low}"; sort ascending by low edge to
    # match the coverage panel's left-to-right voltage ordering.
    feature_bin_cols = sorted(feature_bin_cols, key=lambda c: float(c.split("_")[-1]))
    feature_bin_labels = [f"{c.split('_')[-1]}-{c.split('_')[-2]}" for c in feature_bin_cols]

    bin_edges = np.arange(2.0, 3.6, 0.1)
    bin_labels = [f"{b:.1f}-{b+0.1:.1f}" for b in bin_edges[:-1]]
    x = np.arange(len(bin_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: NMC coverage, real vs. synthetic (raw-trace based) ---
    ax = axes[0]
    real_cov = coverage_by_bin(real_nmc.groupby("Variation_ID"), bin_edges)
    synth_cov = coverage_by_bin(synth_nmc.groupby(GROUPBY_COLS), bin_edges)
    shade_zone(ax, bin_labels, bin_edges)
    ax.bar(x - width / 2, real_cov, width, label="Real NMC", color="tab:blue")
    ax.bar(x + width / 2, synth_cov, width, label="Synthetic NMC (1.5V cutoff)", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of cycles/batteries with a genuine sample in this bin")
    ax.set_title("NMC: per-bin coverage\n(real ~100% vs. synthetic ~0% in the target zone)")
    ax.legend()

    # --- Panel B: LFP magnitude mismatch, real vs. synthetic (from the
    # validated extracted-feature CSV, not a fresh raw recomputation) ---
    ax2 = axes[1]
    real_lfp_mean = features_df[(features_df.DataKind == "real") & (features_df.Chemistry == "LFP")][feature_bin_cols].mean()
    synth_lfp_mean = features_df[(features_df.DataKind == "synthetic") & (features_df.Chemistry == "LFP")][feature_bin_cols].mean()
    xf = np.arange(len(feature_bin_labels))
    shade_zone(ax2, feature_bin_labels, np.append([float(l.split("-")[0]) for l in feature_bin_labels], 3.5))
    ax2.bar(xf - width / 2, real_lfp_mean.values, width, label="Real LFP", color="tab:blue")
    ax2.bar(xf + width / 2, synth_lfp_mean.values, width, label="Synthetic LFP (1.5V cutoff)", color="tab:red")
    ax2.set_xticks(xf)
    ax2.set_xticklabels(feature_bin_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Mean dV/dQ (from extracted features)")
    ax2.set_title("LFP: per-bin dV/dQ magnitude\n(synthetic 2-6x larger than real in the target zone)")
    ax2.legend()

    fig.suptitle("Problem 2 (NOT solved by the cutoff hack): sampling-density and magnitude mismatches, artifact-free",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUT_DIR, "sampling_density_coverage_gap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: simulated vs. real, raw discharge curves + dV/dQ profiles --
# replicates experiment 10's plotting style exactly (see
# experiments/10_constant_current_three/simulate_batteries_const_three.py's
# all-runs discharge plot and feature_engineering_const_three.py's
# plot_dvdq_profiles): #1f77b4/#ff7f0e chemistry colors, thin low-alpha
# individual trajectories with a bold mean overlay, dashed grid at
# alpha=0.5, bold titles, voltage axis inverted so discharge reads left to
# right. Solid/dashed here encodes real vs. synthetic (an analogous reuse
# of exp10's own solid/dashed convention, which encoded chemistry there
# since chemistry is already color-coded in both plots).
# ---------------------------------------------------------------------------

CHEM_COLOR = {"LFP": "#1f77b4", "NMC": "#ff7f0e"}
KIND_STYLE = {"real": {"linestyle": "-", "marker": "o"}, "synthetic": {"linestyle": "--", "marker": "s"}}


def plot_3_sim_vs_real_comparison():
    raw_combined_csv = os.path.join(SCRIPT_DIR, "sim_and_real_raw.csv")
    features_csv = os.path.join(SCRIPT_DIR, "features", "ml_features.csv")
    if not (os.path.exists(raw_combined_csv) and os.path.exists(features_csv)):
        raise FileNotFoundError(
            "Missing sim_and_real_raw.csv or features/ml_features.csv -- "
            "run evaluate_low_voltage_sim_to_real.py first."
        )
    raw = pd.read_csv(raw_combined_csv, low_memory=False)
    raw["Battery_ID"] = raw.groupby(GROUPBY_COLS).ngroup()

    features_df = pd.read_csv(features_csv)
    battery_to_kind = raw.drop_duplicates("Battery_ID").set_index("Battery_ID")["DataKind"]
    features_df["DataKind"] = features_df["Battery_ID"].map(battery_to_kind)
    bin_cols = [c for c in features_df.columns if c.startswith("dV_dQ_V_")]
    v_axis = [(float(c.split("_")[-2]) + float(c.split("_")[-1])) / 2.0 for c in bin_cols]
    sorted_pairs = sorted(zip(v_axis, bin_cols), reverse=True)
    v_axis_sorted = [p[0] for p in sorted_pairs]
    sorted_cols = [p[1] for p in sorted_pairs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # --- Left: raw discharge curves, real vs. synthetic overlaid ---
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

    # --- Right: dV/dQ profiles vs. voltage bin, real vs. synthetic overlaid ---
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
    ax2.set_ylim(-60, 15)
    ax2.axvspan(TARGET_ZONE_MIN, TARGET_ZONE_MAX, color="orange", alpha=0.1, label="Target zone (2.5-3.0V)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=8)

    fig.suptitle("Simulated vs. real data: raw discharge shape and dV/dQ profile divergence (exp10 plotting style)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, "sim_vs_real_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_1_artifact_cutoff_comparison()
    plot_2_sampling_density_coverage_gap()
    plot_3_sim_vs_real_comparison()
