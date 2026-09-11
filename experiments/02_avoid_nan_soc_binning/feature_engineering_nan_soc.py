import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _default_features_dir(input_csv):
    """Sibling 'features/' dir next to input_csv's 'raw/' dir, so feature
    CSVs land in the same data/<run_label>/ tree simulate_batteries.py wrote
    the raw data into, sorted by kind alongside it."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


def create_features_by_voltage_bins(input_csv, step_counts=[5, 10, 15, 20, 25], output_dir=None):
    print(f"Loading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'N_Steps', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()

    all_features = []

    PULSE_DURATION_S = 30 * 60
    REST_DURATION_S = 60 * 60
    STEP_DURATION_S = PULSE_DURATION_S + REST_DURATION_S
    BOUNDARY_TIME_TOL_S = 0.01

    V_BIN_MIN = 1.9
    V_BIN_MAX = 4.3
    V_BIN_WIDTH = 0.1

    for battery_id, group in df.groupby('Battery_ID'):
        group = group.sort_values('Time [s]', kind='stable').reset_index(drop=True)
        n_steps = int(group['N_Steps'].iloc[0])
        times = group['Time [s]'].values

        ocv_rows = []
        for k in range(1, n_steps + 1):
            target_t = k * STEP_DURATION_S
            matches = np.where(np.isclose(times, target_t, atol=BOUNDARY_TIME_TOL_S))[0]
            if matches.size == 0:
                break
            ocv_rows.append(group.iloc[matches[0]])

        if not ocv_rows:
            continue

        ocv_points = pd.DataFrame(ocv_rows).reset_index(drop=True)
        start_row = group.iloc[[0]][['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]
        ocv_sequence = pd.concat([start_row, ocv_points[['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]], ignore_index=True)

        dV = ocv_sequence['Voltage [V]'].diff()
        dQ = ocv_sequence['Capacity [A.h]'].diff()
        valid = (dQ > 1e-5)

        valid_tail = valid.values[1:]
        invalid_positions = np.where(~valid_tail)[0]
        if invalid_positions.size > 0:
            first_invalid = invalid_positions[0]
            assert valid_tail[first_invalid:].sum() == 0, (
                f"Battery_ID {battery_id}: found a valid dV/dQ transition after an "
                f"invalid one (first invalid at pulse {first_invalid + 1})."
            )

        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Target_Capacity_Ah': group['Target_Capacity_Ah'].iloc[0],
            'Size_Multiplier': group['Size_Multiplier'].iloc[0],
            'SOH': group['SOH'].iloc[0],
            'Initial_SOC': group['Initial_SOC'].iloc[0],
            'N_Steps': int(group['N_Steps'].iloc[0]),
        }
        for optional_col in ('Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set'):
            if optional_col in group.columns:
                battery_features[optional_col] = group[optional_col].iloc[0]

        n_v_bins = round((V_BIN_MAX - V_BIN_MIN) / V_BIN_WIDTH)
        voltage_bins = [
            f"dV_dQ_V_{round(V_BIN_MIN + i * V_BIN_WIDTH, 1):.1f}_{round(V_BIN_MIN + (i - 1) * V_BIN_WIDTH, 1):.1f}"
            for i in range(n_v_bins, 0, -1)
        ]
        for bin_name in voltage_bins:
            battery_features[bin_name] = np.nan

        for idx in dV[valid].index:
            dv_val = dV.loc[idx]
            dq_val = dQ.loc[idx]
            dvdq = dv_val / dq_val

            v_val = ocv_sequence['Voltage [V]'].iloc[idx]
            bin_high = np.ceil(round(v_val, 4) * 10) / 10.0
            bin_high = min(V_BIN_MAX, max(V_BIN_MIN + V_BIN_WIDTH, bin_high))
            bin_high = round(bin_high, 1)
            bin_low = round(bin_high - V_BIN_WIDTH, 1)

            bin_key = f"dV_dQ_V_{bin_high:.1f}_{bin_low:.1f}"
            battery_features[bin_key] = dvdq

        all_features.append(battery_features)

    full_df = pd.DataFrame(all_features)

    datasets = {}
    print("\nExtraction Summary (Voltage Bins):")
    for n in step_counts:
        step_subset = full_df[full_df['N_Steps'] == n].copy()

        all_v_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]
        empty_v_cols = [c for c in all_v_cols if step_subset[c].isna().all()]
        step_subset = step_subset.drop(columns=empty_v_cols)

        feat_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]

        out_name = os.path.join(output_dir, f"ml_features_{n}_steps_vbins.csv")
        step_subset.to_csv(out_name, index=False)
        datasets[n] = step_subset

        valid_vals = step_subset[feat_cols].notna().sum().sum()
        print(f"  -> {out_name}: {len(step_subset)} batteries, {len(feat_cols)} voltage bin columns, {valid_vals} valid entries")

    return datasets


def create_features_by_soc_bins(input_csv, step_counts=[5, 10, 15, 20, 25], target_soc_min=0.2, target_soc_max=0.4, output_dir=None):
    print(f"\nLoading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'N_Steps', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()

    all_features = []

    PULSE_DURATION_S = 30 * 60
    REST_DURATION_S = 60 * 60
    STEP_DURATION_S = PULSE_DURATION_S + REST_DURATION_S
    BOUNDARY_TIME_TOL_S = 0.01

    for battery_id, group in df.groupby('Battery_ID'):
        group = group.sort_values('Time [s]', kind='stable').reset_index(drop=True)
        n_steps = int(group['N_Steps'].iloc[0])
        times = group['Time [s]'].values

        ocv_rows = []
        for k in range(1, n_steps + 1):
            target_t = k * STEP_DURATION_S
            matches = np.where(np.isclose(times, target_t, atol=BOUNDARY_TIME_TOL_S))[0]
            if matches.size == 0:
                break
            ocv_rows.append(group.iloc[matches[0]])

        if not ocv_rows:
            continue

        ocv_points = pd.DataFrame(ocv_rows).reset_index(drop=True)
        start_row = group.iloc[[0]][['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]
        ocv_sequence = pd.concat([start_row, ocv_points[['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]], ignore_index=True)

        dV = ocv_sequence['Voltage [V]'].diff()
        dQ = ocv_sequence['Capacity [A.h]'].diff()
        valid = (dQ > 1e-5)

        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Target_Capacity_Ah': group['Target_Capacity_Ah'].iloc[0],
            'Size_Multiplier': group['Size_Multiplier'].iloc[0],
            'SOH': group['SOH'].iloc[0],
            'Initial_SOC': group['Initial_SOC'].iloc[0],
            'N_Steps': int(group['N_Steps'].iloc[0]),
        }
        for optional_col in ('Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set'):
            if optional_col in group.columns:
                battery_features[optional_col] = group[optional_col].iloc[0]

        soc_bins = [f"dV_dQ_SOC_{round(i / 10, 1):.1f}_{round((i - 1) / 10, 1):.1f}" for i in range(10, 0, -1)]
        for bin_name in soc_bins:
            battery_features[bin_name] = np.nan

        nominal_capacity = group['Target_Capacity_Ah'].iloc[0]
        init_soc = group['Initial_SOC'].iloc[0]
        for idx in dV[valid].index:
            dv_val = dV.loc[idx]
            dq_val = dQ.loc[idx]
            dvdq = dv_val / dq_val

            cum_capacity = ocv_sequence['Capacity [A.h]'].iloc[idx]
            current_soc = max(0.0, init_soc - (cum_capacity / nominal_capacity))

            bin_high = min(1.0, np.ceil(round(current_soc, 4) * 10) / 10.0)
            if bin_high <= 0.0:
                bin_high = 0.1
            bin_low = round(bin_high - 0.1, 1)

            bin_key = f"dV_dQ_SOC_{bin_high:.1f}_{bin_low:.1f}"
            battery_features[bin_key] = dvdq

        all_features.append(battery_features)

    full_df = pd.DataFrame(all_features)

    target_bins = []
    curr_high = target_soc_max
    while curr_high > target_soc_min:
        curr_low = round(curr_high - 0.1, 1)
        target_bins.append(f"dV_dQ_SOC_{curr_high:.1f}_{curr_low:.1f}")
        curr_high = curr_low

    datasets = {}
    print(f"Extraction Summary (Target SOC Window: {target_soc_min} to {target_soc_max}):")
    for n in step_counts:
        step_subset = full_df[full_df['N_Steps'] == n].copy()

        valid_rows_mask = step_subset[target_bins].notna().all(axis=1)
        step_subset = step_subset[valid_rows_mask].copy()

        non_feat_cols = [c for c in step_subset.columns if not c.startswith('dV_dQ_SOC_')]
        step_subset = step_subset[non_feat_cols + target_bins]

        out_name = os.path.join(output_dir, f"ml_features_{n}_steps_socbins.csv")
        step_subset.to_csv(out_name, index=False)
        datasets[n] = step_subset

        print(f"  -> {out_name}: {len(step_subset)} batteries, {len(target_bins)} SOC bin columns, {step_subset[target_bins].notna().sum().sum()} valid entries")

    return datasets


def plot_all_voltage_bin_profiles(datasets):
    step_counts = sorted(datasets.keys())
    fig, axes = plt.subplots(len(step_counts), 1, figsize=(10, 3.5 * len(step_counts)))

    if len(step_counts) == 1:
        axes = [axes]

    for ax, n_steps in zip(axes, step_counts):
        subset = datasets[n_steps]

        lfp_rows = subset[subset['Chemistry'] == 'LFP']
        nmc_rows = subset[subset['Chemistry'] == 'NMC']

        if lfp_rows.empty or nmc_rows.empty:
            ax.set_title(f"{n_steps} Steps - Missing LFP or NMC data")
            continue

        lfp_sample = lfp_rows.iloc[0]
        nmc_sample = nmc_rows.iloc[0]

        feature_cols = [c for c in subset.columns if c.startswith('dV_dQ_V_')]

        y_lfp = lfp_sample[feature_cols].astype(float).dropna()
        y_nmc = nmc_sample[feature_cols].astype(float).dropna()

        x_lfp = [float(col.split('_')[-2]) for col in y_lfp.index]
        x_nmc = [float(col.split('_')[-2]) for col in y_nmc.index]

        ax.plot(x_lfp, y_lfp.values, marker='o', label='LFP', color='#1f77b4', linewidth=2, markersize=6)
        ax.plot(x_nmc, y_nmc.values, marker='s', label='NMC', color='#ff7f0e', linewidth=2, markersize=6)

        ax.set_title(f'GITT Pulse Profile: {n_steps} Steps (Absolute voltage bins)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Terminal Voltage [V]')
        ax.set_ylabel('dV/dQ [V/Ah]')

        all_x = sorted(set(x_lfp + x_nmc), reverse=True)
        if all_x:
            ax.set_xticks(all_x)
        ax.invert_xaxis()

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_all_soc_bin_profiles(datasets):
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 3 * len(datasets)), sharex=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (n_steps, df_subset) in zip(axes, datasets.items()):
        dvdq_cols = [c for c in df_subset.columns if c.startswith('dV_dQ_SOC_')]
        if not dvdq_cols or df_subset.empty:
            ax.set_title(f"GITT Pulse Profile: {n_steps} Steps (No valid SOC bins)")
            continue

        plot_data = df_subset.melt(
            id_vars=['Chemistry'],
            value_vars=dvdq_cols,
            var_name='SOC_Bin',
            value_name='dV_dQ'
        )

        sns.lineplot(
            data=plot_data,
            x='SOC_Bin',
            y='dV_dQ',
            hue='Chemistry',
            marker='o',
            ax=ax
        )

        ax.set_title(f"GITT Pulse Profile: {n_steps} Steps (Target SOC Bins)")
        ax.set_ylabel("dV/dQ [V/Ah]")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.xlabel("State of Charge Bin")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_file = os.path.join("data", "default", "raw", "advanced_synthetic_battery_data.csv")
    step_list = [5, 10, 15, 20, 25]

    # 1. Extract & plot voltage-binned features
    v_datasets = create_features_by_voltage_bins(input_file, step_counts=step_list)
    plot_all_voltage_bin_profiles(v_datasets)

    # 2. Extract & plot SOC-binned features (target window: 0.2 to 0.4)
    soc_datasets = create_features_by_soc_bins(
        input_file,
        step_counts=step_list,
        target_soc_min=0.2,
        target_soc_max=0.4
    )
    plot_all_soc_bin_profiles(soc_datasets)
