import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

#Determines where to store the extracted features relative to the input CSV file 
def _default_features_dir(input_csv):
    """Sibling 'features/' dir next to input_csv's 'raw/' dir."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


"""FILE LOADING AND FEATURE EXTRACTION
Assigns a single numeric battery id to each continuous discharge run. 
Grouping on variation id prevent independaent runs with identical rounded values from merging"""
def create_features_by_voltage_bins_continuous(input_csv, output_dir=None):
    print(f"Loading raw continuous simulation data from '{input_csv}'...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found at: {input_csv}")

    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Unique identifier per continuous discharge run
    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()

    # Voltage grid spanning full ranges of both LFP (down to 2.0 V) and NMC (up to 4.3 V)
    # This ensures every battery, regardless of duration or starting point, is evaluated
    # against the exact same uniform column template
    V_BIN_MIN = 1.5
    V_BIN_MAX = 4.3
    V_BIN_WIDTH = 0.1
    n_v_bins = int(round((V_BIN_MAX - V_BIN_MIN) / V_BIN_WIDTH))
    voltage_bins = [
        f"dV_dQ_V_{round(V_BIN_MIN + i * V_BIN_WIDTH, 1):.1f}_{round(V_BIN_MIN + (i - 1) * V_BIN_WIDTH, 1):.1f}"
        for i in range(n_v_bins, 0, -1)
    ]

    all_features = []

    for battery_id, group in df.groupby('Battery_ID'):
        group = group.sort_values('Time [s]', kind='stable').reset_index(drop=True)

        if len(group) < 15: #skips corrupted or immediatly aborted runs
            continue

        raw_v = group['Voltage [V]'].values
        raw_q = group['Capacity [A.h]'].values

        # 1. Smooth sensor noise to prevent division-by-tiny-dQ explosions
        #Savizky-Golay filter
        window_len = min(21, len(raw_v) if len(raw_v) % 2 != 0 else len(raw_v) - 1)
        if window_len >= 5:
            v_smooth = savgol_filter(raw_v, window_length=window_len, polyorder=2)
        else:
            v_smooth = raw_v

        # 2. Derivative calculation
        dV = np.diff(v_smooth)
        dQ = np.diff(raw_q)
        valid = dQ > 1e-5 #prevents division by zero errors during resting or static intervals

        if not np.any(valid):
            continue

        #metadata storage
        dvdq_vals = dV[valid] / dQ[valid]
        v_eval_pts = v_smooth[1:][valid]

        # Base metadata
        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Target_Capacity_Ah': group['Target_Capacity_Ah'].iloc[0],
            'Size_Multiplier': group['Size_Multiplier'].iloc[0],
            'SOH': group['SOH'].iloc[0],
            'Initial_SOC': group['Initial_SOC'].iloc[0],
            'Variation_ID': group['Variation_ID'].iloc[0],
        }

        # Keep runtime operational variables
        optional_cols = (
            'C_Rate',
            'Current [A]',
            'Ambient_Temperature_C',
            'Resistance_Factor',
            'Base_Parameter_Set',
        )
        for col in optional_cols:
            if col in group.columns:
                battery_features[col] = group[col].iloc[0]

        # 3. Bin allocation: Each point to point transition is mapped into
        # its corresponding 0.1V window based on its instantaneous voltage
        bin_values = {}
        for dvdq, v_val in zip(dvdq_vals, v_eval_pts):
            if v_val < V_BIN_MIN or v_val > V_BIN_MAX:
                continue

            bin_idx = int(np.floor(round((v_val - V_BIN_MIN) / V_BIN_WIDTH, 4)))
            bin_idx = min(n_v_bins - 1, max(0, bin_idx))

            bin_low = round(V_BIN_MIN + bin_idx * V_BIN_WIDTH, 1)
            bin_high = round(bin_low + V_BIN_WIDTH, 1)
            bin_key = f"dV_dQ_V_{bin_high:.1f}_{bin_low:.1f}"

            bin_values.setdefault(bin_key, []).append(dvdq)

        # Average dV/dQ values within each decile bin; absent bins remain NaN
        # Multiple measurements points falling into the same 0.1V window are averaged
        # into a single representetive dvdq value. 
        for bin_name in voltage_bins:
            vals = bin_values.get(bin_name)
            battery_features[bin_name] = float(np.mean(vals)) if vals else np.nan

        all_features.append(battery_features)

    full_df = pd.DataFrame(all_features)

    # Drop bins that no battery ever reached across the entire dataset
    all_v_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]
    empty_v_cols = [c for c in all_v_cols if full_df[c].isna().all()]
    full_df = full_df.drop(columns=empty_v_cols)

    feat_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]

    out_name = os.path.join(output_dir, "ml_features_continuous.csv")
    full_df.to_csv(out_name, index=False)

    valid_entries = full_df[feat_cols].notna().sum().sum()
    print(f"\nExtraction complete:")
    print(f" -> Saved to: '{out_name}'")
    print(f" -> Batteries processed: {len(full_df)}")
    print(f" -> Active voltage bins: {len(feat_cols)}")
    print(f" -> Total valid feature entries: {valid_entries}")

    return full_df, output_dir

"""PLOTTING
Plots every individual run with high transparancy, 
highlighting operational variance across C-rates and aging states
Overlays bold mean curves for each chemistry to visualize the LFP plateau vs NMC slope
"""
def plot_dvdq_profiles(features_df, output_dir=None, filename="dvdq_profiles_plot.png"):
    """Plots binned dV/dQ profiles against voltage for LFP vs NMC."""
    feat_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    if not feat_cols:
        print("No dV/dQ feature columns found to plot.")
        return

    # Extract midpoint voltage from bin names: 'dV_dQ_V_3.4_3.3' -> (3.4 + 3.3) / 2 = 3.35 V
    voltage_points = []
    for col in feat_cols:
        parts = col.split('_')
        v_high = float(parts[-2])
        v_low = float(parts[-1])
        voltage_points.append((v_high + v_low) / 2.0)

    # Sort columns by voltage descending (discharge direction: 4.2 V down to 2.0 V)
    sorted_pairs = sorted(zip(voltage_points, feat_cols), reverse=True)
    v_axis = [p[0] for p in sorted_pairs]
    sorted_cols = [p[1] for p in sorted_pairs]

    lfp_data = features_df[features_df['Chemistry'] == 'LFP']
    nmc_data = features_df[features_df['Chemistry'] == 'NMC']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual battery trajectories with alpha transparency
    for _, row in lfp_data.iterrows():
        y_vals = row[sorted_cols].astype(float).values
        ax.plot(v_axis, y_vals, color='#1f77b4', alpha=0.08, linewidth=0.8)

    for _, row in nmc_data.iterrows():
        y_vals = row[sorted_cols].astype(float).values
        ax.plot(v_axis, y_vals, color='#ff7f0e', alpha=0.08, linewidth=0.8)

    # Plot bold mean curves per chemistry
    lfp_mean = lfp_data[sorted_cols].mean(skipna=True).values
    nmc_mean = nmc_data[sorted_cols].mean(skipna=True).values

    ax.plot(v_axis, lfp_mean, marker='o', color='#1f77b4', linewidth=2.5, markersize=5, label='LFP (Mean Profile)')
    ax.plot(v_axis, nmc_mean, marker='s', color='#ff7f0e', linewidth=2.5, markersize=5, label='NMC (Mean Profile)')

    ax.set_title("Extracted Continuous dV/dQ Feature Profiles across Voltage Bins", fontsize=12, fontweight='bold')
    ax.set_xlabel("Cell Terminal Voltage [V] (Bin Center)", fontsize=11)
    ax.set_ylabel("dV/dQ [V/Ah]", fontsize=11)

    # Invert x-axis so discharge progresses left-to-right (high voltage to low voltage)
    ax.invert_xaxis()

    # Clip y-axis display range so asymptotic end-of-discharge drops do not compress the main profile
    ax.set_ylim(-30, 2)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if output_dir:
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=150)
        print(f" -> dV/dQ profile plot saved to: '{plot_path}'")

    plt.close(fig)


if __name__ == "__main__":
    input_file = os.path.join(
        "data", "continuous_discharge", "raw", "continuous_synthetic_battery_data.csv"
    )
    features_df, out_dir = create_features_by_voltage_bins_continuous(input_file)
    plot_dvdq_profiles(features_df, output_dir=out_dir)