"""
Overlays all differential capacity (signed dV/dQ) curves on a single plot:
- LFP Synthetic vs. LFP Real
- NMC Synthetic vs. NMC Real
Directly compares chemistry separation and sim-to-real alignment across voltage bins.
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(SCRIPT_DIR, "features", "ml_features_ELLA.csv")
RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw_ELLA.csv")
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, "plots", "dvdq_all_in_one_comparison.png")

GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
BIN_RE = re.compile(r"dV_dQ_V_([\d.]+)_([\d.]+)")


def load_features():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"'{FEATURES_CSV}' not found. Run evaluate_empa_soc_sweep.py first.")

    features_df = pd.read_csv(FEATURES_CSV)

    if 'DataKind' not in features_df.columns:
        raw_df = pd.read_csv(RAW_CSV, usecols=GROUPBY_COLS + ['DataKind'], low_memory=False)
        raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
        id_to_kind = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
        features_df['DataKind'] = features_df['Battery_ID'].map(id_to_kind)

    bin_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    bin_cols = sorted(bin_cols, key=lambda c: -float(c.split('_')[3]))
    return features_df, bin_cols


def parse_bin_midpoints(bin_cols):
    midpoints = []
    for col in bin_cols:
        m = BIN_RE.match(col)
        midpoints.append((float(m.group(1)) + float(m.group(2))) / 2.0)
    return np.array(midpoints)


def main():
    features_df, bin_cols = load_features()
    v_midpoints = parse_bin_midpoints(bin_cols)

    plt.figure(figsize=(12, 7))

    # Distinct color palette: Blue for LFP, Orange/Red for NMC
    curve_configs = {
        ('LFP', 'synthetic'): {
            'label': 'LFP - Synthetic (PyBaMM)',
            'color': '#1f77b4',
            'linestyle': '-',
            'marker': 'o',
        },
        ('LFP', 'real'): {
            'label': 'LFP - Real (EMPA)',
            'color': '#004c6d',
            'linestyle': '--',
            'marker': 's',
        },
        ('NMC', 'synthetic'): {
            'label': 'NMC - Synthetic (PyBaMM)',
            'color': '#ff7f0e',
            'linestyle': '-',
            'marker': '^',
        },
        ('NMC', 'real'): {
            'label': 'NMC - Real (EMPA)',
            'color': '#d62728',
            'linestyle': '--',
            'marker': 'v',
        },
    }

    for (chem, kind), cfg in curve_configs.items():
        subset = features_df[(features_df['Chemistry'] == chem) & (features_df['DataKind'] == kind)]
        if len(subset) == 0:
            continue

        # Keep signed negative values
        matrix = subset[bin_cols].replace([np.inf, -np.inf], np.nan)

        mean_vals = matrix.mean(axis=0).values
        std_vals = matrix.std(axis=0).values

        valid_mask = ~np.isnan(mean_vals)
        x = v_midpoints[valid_mask]
        y = mean_vals[valid_mask]
        s = std_vals[valid_mask]

        # Plot main curve
        plt.plot(
            x, y,
            label=f"{cfg['label']} (n={len(subset)})",
            color=cfg['color'],
            linestyle=cfg['linestyle'],
            marker=cfg['marker'],
            markersize=5,
            linewidth=2.2,
        )

        # Shaded variability band (±0.5 std for clarity when all 4 are overlaid)
        plt.fill_between(x, y - 0.5 * s, y + 0.5 * s, color=cfg['color'], alpha=0.10)

    plt.title("LFP vs. NMC: Sim-to-Real Differential Capacity Overlay", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Terminal Voltage Bin Midpoint [V]", fontsize=12)
    plt.ylabel("Mean Signed dV/dQ [V / A·h]", fontsize=12)

    # Invert x-axis (discharge direction: 3.3V down to 2.5V)
    plt.xlim(max(v_midpoints) + 0.05, min(v_midpoints) - 0.05)
    plt.ylim(-10.0, 0.5)

    # Add a horizontal zero line and subtle grid
    plt.axhline(0, color='black', linewidth=0.8, linestyle=':')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", framealpha=0.92, fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Saved combined overlay plot to '{OUTPUT_PLOT}'")


if __name__ == "__main__":
    main()