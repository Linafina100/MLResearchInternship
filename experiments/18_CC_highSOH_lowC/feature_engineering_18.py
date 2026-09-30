import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Determines where to store the extracted features relative to the input CSV file  
def _default_features_dir(input_csv):
    """Sibling 'features/' dir next to input_csv's 'raw/' dir."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


"""FILE LOADING AND FEATURE EXTRACTION
Assigns a single numeric battery id to each continuous discharge run. 
Grouping on variation id prevents independent runs with identical rounded values from merging.
"""
def create_features_by_voltage_bins_continuous(
    input_csv,
    output_dir=None,
    output_filename="ml_features_continuous_18.csv",
    min_chemistry_coverage=0.2,
):
    print(f"\nLoading raw continuous discharge data from '{input_csv}'...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found at: {input_csv}")

    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Standardize column headers
    col_mapping = {
        "Capacity [Ah]": "Capacity [A.h]",
        "Discharged Capacity [Ah]": "Capacity [A.h]",
        "Terminal Voltage [V]": "Voltage [V]",
    }
    df.rename(columns=col_mapping, inplace=True)
    df["Chemistry"] = df["Chemistry"].astype(str).str.strip().str.upper()

    # Unique identifier per continuous discharge run
    groupby_cols = [c for c in ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID'] if c in df.columns]
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()

    # Voltage grid spanning full ranges of both LFP (down to 1.5 V) and NMC (up to 4.3 V)
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
        sort_col = 'Time [s]' if 'Time [s]' in group.columns else 'Capacity [A.h]'
        group = group.sort_values(sort_col, kind='stable').reset_index(drop=True)

        if len(group) < 15:  # skips corrupted or immediately aborted runs
            continue

        raw_v = group['Voltage [V]'].values
        raw_q = group['Capacity [A.h]'].values

        # Ensure strictly increasing capacity for interpolation
        unique_q, unique_idx = np.unique(raw_q, return_index=True)
        unique_v = raw_v[unique_idx]

        if len(unique_q) < 10 or (unique_q[-1] - unique_q[0]) < 0.05:
            continue

        # 1. Resample to a uniform capacity grid to prevent non-uniform derivative noise
        q_grid = np.linspace(unique_q[0], unique_q[-1], 200)
        v_interp = np.interp(q_grid, unique_q, unique_v)

        # 2. Savitzky-Golay filtering across equidistant points
        v_smooth = savgol_filter(v_interp, window_length=17, polyorder=2)

        # 3. Finite difference calculation
        dV = np.diff(v_smooth)
        dQ = np.diff(q_grid)
        dvdq_vals = dV / dQ
        v_eval_pts = v_smooth[1:]

        # Base metadata
        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Target_Capacity_Ah': group['Target_Capacity_Ah'].iloc[0] if 'Target_Capacity_Ah' in group.columns else 0.0,
            'Size_Multiplier': group['Size_Multiplier'].iloc[0] if 'Size_Multiplier' in group.columns else 1.0,
            'SOH': group['SOH'].iloc[0] if 'SOH' in group.columns else 1.0,
            'Initial_SOC': group['Initial_SOC'].iloc[0] if 'Initial_SOC' in group.columns else 1.0,
            'Variation_ID': group['Variation_ID'].iloc[0],
        }

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

        # 4. Bin allocation
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

        for bin_name in voltage_bins:
            vals = bin_values.get(bin_name)
            battery_features[bin_name] = float(np.mean(vals)) if vals else np.nan

        all_features.append(battery_features)

    full_df = pd.DataFrame(all_features)

    # Drop bins completely empty across all batteries
    all_v_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]
    empty_v_cols = [c for c in all_v_cols if full_df[c].isna().all()]
    full_df = full_df.drop(columns=empty_v_cols)

    feat_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]

    # Drop one-sided bins if multiple chemistries are present
    if full_df['Chemistry'].nunique() > 1:
        coverage_by_chem = full_df.groupby('Chemistry')[feat_cols].apply(lambda g: g.notna().mean())
        one_sided_cols = [c for c in feat_cols if (coverage_by_chem[c] < min_chemistry_coverage).any()]
        full_df = full_df.drop(columns=one_sided_cols)
        feat_cols = [c for c in feat_cols if c not in one_sided_cols]

    out_name = os.path.join(output_dir, output_filename)
    full_df.to_csv(out_name, index=False)

    valid_entries = full_df[feat_cols].notna().sum().sum()
    print(f"Extraction complete:")
    print(f" -> Saved to: '{out_name}'")
    print(f" -> Batteries processed: {len(full_df)}")
    print(f" -> Active voltage bins: {len(feat_cols)}")
    print(f" -> Total valid feature entries: {valid_entries}")

    return full_df, output_dir


"""PLOTTING"""
def plot_dvdq_profiles(features_df, output_dir=None, filename="dvdq_profiles_plot_18.png"):
    feat_cols = [c for c in features_df.columns if c.startswith('dV_dQ_V_')]
    if not feat_cols:
        print("No dV/dQ feature columns found to plot.")
        return

    voltage_points = []
    for col in feat_cols:
        parts = col.split('_')
        v_high = float(parts[-2])
        v_low = float(parts[-1])
        voltage_points.append((v_high + v_low) / 2.0)

    sorted_pairs = sorted(zip(voltage_points, feat_cols), reverse=True)
    v_axis = [p[0] for p in sorted_pairs]
    sorted_cols = [p[1] for p in sorted_pairs]

    lfp_data = features_df[features_df['Chemistry'] == 'LFP']
    nmc_data = features_df[features_df['Chemistry'] == 'NMC']

    fig, ax = plt.subplots(figsize=(12, 6))

    for _, row in lfp_data.iterrows():
        y_vals = row[sorted_cols].astype(float).values
        ax.plot(v_axis, y_vals, color='#1f77b4', alpha=0.08, linewidth=0.8)

    for _, row in nmc_data.iterrows():
        y_vals = row[sorted_cols].astype(float).values
        ax.plot(v_axis, y_vals, color='#ff7f0e', alpha=0.08, linewidth=0.8)

    lfp_mean = lfp_data[sorted_cols].mean(skipna=True).values
    nmc_mean = nmc_data[sorted_cols].mean(skipna=True).values

    if len(lfp_data) > 0:
        ax.plot(v_axis, lfp_mean, marker='o', color='#1f77b4', linewidth=2.5, markersize=5, label='LFP (Mean Profile)')
    if len(nmc_data) > 0:
        ax.plot(v_axis, nmc_mean, marker='s', color='#ff7f0e', linewidth=2.5, markersize=5, label='NMC (Mean Profile)')

    ax.set_title("Extracted Continuous dV/dQ Feature Profiles across Voltage Bins", fontsize=12, fontweight='bold')
    ax.set_xlabel("Cell Terminal Voltage [V] (Bin Center)", fontsize=11)
    ax.set_ylabel("dV/dQ [V/Ah]", fontsize=11)
    ax.invert_xaxis()
    ax.set_ylim(-30, 2)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if output_dir:
        if os.path.basename(os.path.normpath(output_dir)) == "features":
            plot_dir = os.path.join(os.path.dirname(output_dir), "plots")
        else:
            plot_dir = output_dir
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, filename)
        plt.savefig(plot_path, dpi=150)
        print(f" -> dV/dQ profile plot saved to: '{plot_path}'")

    plt.close(fig)


if __name__ == "__main__":
    DATA_DIR = os.environ.get("DATA_DIR", "data")
    RUN_LABEL = os.environ.get("RUN_LABEL", "continuous_discharge_18")
    RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)
    
    RAW_DIR = os.path.join(RUN_DIR, "raw")
    FEATURES_DIR = os.path.join(RUN_DIR, "features")

    # 1. Process synthetic dataset -> ml_features_continuous_18.csv
    synth_input_file = os.environ.get(
        "OUTPUT_DATA_CSV",
        os.path.join(RAW_DIR, "constant_current_synthetic_battery_data_18.csv"),
    )
    if os.path.exists(synth_input_file):
        synth_df, _ = create_features_by_voltage_bins_continuous(
            synth_input_file,
            output_dir=FEATURES_DIR,
            output_filename="ml_features_continuous_18.csv",
        )
        plot_dvdq_profiles(synth_df, output_dir=FEATURES_DIR, filename="dvdq_profiles_plot_18.png")

    # 2. Process real experimental dataset -> ml_features_real_18.csv
    real_input_file = os.environ.get(
        "REAL_DATA_CSV",
        os.path.join(RAW_DIR, "real_combined_raw.csv"),
    )
    if os.path.exists(real_input_file):
        real_df, _ = create_features_by_voltage_bins_continuous(
            real_input_file,
            output_dir=FEATURES_DIR,
            output_filename="ml_features_real_18.csv",
        )
        plot_dvdq_profiles(real_df, output_dir=FEATURES_DIR, filename="dvdq_profiles_real_18.png")
    else:
        print(f"\nNote: Real data file not found at '{real_input_file}'. Skipping real feature extraction.")