# Archived snapshot of the GITT-pulse-boundary feature engineering: this
# was the repo-root feature_engineering.py until the continuous-discharge
# variant was promoted to root in its place. Still used by
# experiments/01_.../soc_sweep.py and
# experiments/04_.../sweep_pulse_variable_discharge.py, which pair it with
# the pulse protocol archived alongside it in this folder.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- FILE STRUCTURE ---

def _default_features_dir(input_csv):
    """Sibling 'features/' dir next to input_csv's 'raw/' dir, so feature
    CSVs land in the same data/<run_label>/ tree simulate_batteries.py wrote
    the raw data into, sorted by kind alongside it."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


def create_features_by_voltage_bins(input_csv, step_counts=[5, 10, 15, 20, 25], output_dir=None,
                                     min_chemistry_coverage=0.2):
    print(f"Loading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'N_Steps', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()
    
    all_features = []
    
    # Identify step boundaries by exact timestamps (30m pulse + 60m rest).
    PULSE_DURATION_S = 30 * 60   # matches PULSE_DURATION in simulate_batteries.py
    REST_DURATION_S = 60 * 60    # matches REST_DURATION in simulate_batteries.py
    STEP_DURATION_S = PULSE_DURATION_S + REST_DURATION_S
    BOUNDARY_TIME_TOL_S = 0.01   # error range

    # Use absolute voltage bins.
    V_BIN_MIN = 1.9
    V_BIN_MAX = 4.3
    V_BIN_WIDTH = 0.1

    for battery_id, group in df.groupby('Battery_ID'):
        # Sort chronologically.
        group = group.sort_values('Time [s]', kind='stable').reset_index(drop=True)
        n_steps = int(group['N_Steps'].iloc[0])
        times = group['Time [s]'].values

        ocv_rows = []
        for k in range(1, n_steps + 1):
            target_t = k * STEP_DURATION_S
            matches = np.where(np.isclose(times, target_t, atol=BOUNDARY_TIME_TOL_S))[0]
            if matches.size == 0:
                break  # this step (and any later ones) never completed - early voltage cutoff
            ocv_rows.append(group.iloc[matches[0]])  # first = still-relaxed value

        if not ocv_rows:
            continue

        ocv_points = pd.DataFrame(ocv_rows).reset_index(drop=True)

        # Include the initial starting point (t=0) as the reference OCV
        start_row = group.iloc[[0]][['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]
        ocv_sequence = pd.concat([start_row, ocv_points[['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]], ignore_index=True)
        
        # Calculate dV and dQ between consecutive relaxed points
        dV = ocv_sequence['Voltage [V]'].diff()
        dQ = ocv_sequence['Capacity [A.h]'].diff()     # Discharged capacity

        # Keep only valid intervals where capacity actually advanced
        valid = (dQ > 1e-5)

        # Ensure that if a battery hits a voltage cutoff and aborts a pulse, 
        # the failure only occurs at the very end of the test sequence. 
        # A mid-sequence failure followed by a "valid" step would break positional indexing
        valid_tail = valid.values[1:]
        invalid_positions = np.where(~valid_tail)[0]
        if invalid_positions.size > 0:
            first_invalid = invalid_positions[0]
            assert valid_tail[first_invalid:].sum() == 0, (
                f"Battery_ID {battery_id}: found a valid dV/dQ transition after an "
                f"invalid one (first invalid at pulse {first_invalid + 1}). Expected "
                "invalid transitions only at the end of the sequence (early voltage "
                "cutoff) — a mid-sequence gap would break positional step indexing."
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
        # Carried through for analysis only
        for optional_col in ('Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set'):
            if optional_col in group.columns:
                battery_features[optional_col] = group[optional_col].iloc[0]

        # Pre-initialize standard voltage bins (descending, matching discharge
        # direction) so every battery gets the same uniform column set.
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

            # Terminal voltage at this OCV point
            v_val = ocv_sequence['Voltage [V]'].iloc[idx]
            bin_high = np.ceil(round(v_val, 4) * 10) / 10.0
            bin_high = min(V_BIN_MAX, max(V_BIN_MIN + V_BIN_WIDTH, bin_high))
            bin_high = round(bin_high, 1)
            bin_low = round(bin_high - V_BIN_WIDTH, 1)

            bin_key = f"dV_dQ_V_{bin_high:.1f}_{bin_low:.1f}"
            battery_features[bin_key] = dvdq

        all_features.append(battery_features)
        
    full_df = pd.DataFrame(all_features)
    
    # Split into 5 distinct datasets corresponding to each step count (5, 10, 15, 20, 25)
    datasets = {}
    print("\nExtraction Summary:")
    for n in step_counts:
        step_subset = full_df[full_df['N_Steps'] == n].copy()

        # Drop bins entirely unreached at this step count so feat_cols
        # matches the model's real trained width.
        all_v_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]
        empty_v_cols = [c for c in all_v_cols if step_subset[c].isna().all()]
        step_subset = step_subset.drop(columns=empty_v_cols)

        # Find all valid voltage bin columns for this step count
        feat_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]

        # Drop bins only one chemistry ever reaches: SimpleImputer's
        # median-fill would otherwise give every row of the missing
        # chemistry an identical constant -- a trivial giveaway, not real
        # signal. Full writeup: experiments/01_leakage_fix_20_percent_coverage/RESULTS.md.
        coverage_by_chem = step_subset.groupby('Chemistry')[feat_cols].apply(lambda g: g.notna().mean())
        one_sided_cols = [c for c in feat_cols if (coverage_by_chem[c] < min_chemistry_coverage).any()]
        step_subset = step_subset.drop(columns=one_sided_cols)
        feat_cols = [c for c in feat_cols if c not in one_sided_cols]

        out_name = os.path.join(output_dir, f"ml_features_{n}_steps.csv")
        step_subset.to_csv(out_name, index=False)
        datasets[n] = step_subset

        # Sanity check: count non-null values
        valid_vals = step_subset[feat_cols].notna().sum().sum()
        print(f"  -> {out_name}: {len(step_subset)} batteries, {len(feat_cols)} voltage bin columns, {valid_vals} valid entries")

    return datasets


def plot_all_voltage_bin_profiles(datasets):
    step_counts = sorted(datasets.keys())
    fig, axes = plt.subplots(len(step_counts), 1, figsize=(10, 3.5 * len(step_counts)))
    
    if len(step_counts) == 1:
        axes = [axes]

    for ax, n_steps in zip(axes, step_counts):
        subset = datasets[n_steps]
        
        # Separate chemistries
        lfp_rows = subset[subset['Chemistry'] == 'LFP']
        nmc_rows = subset[subset['Chemistry'] == 'NMC']
        
        if lfp_rows.empty or nmc_rows.empty:
            ax.set_title(f"{n_steps} Steps - Missing LFP or NMC data")
            continue

        lfp_sample = lfp_rows.iloc[0]
        nmc_sample = nmc_rows.iloc[0]
        
        # Extract features for this step
        feature_cols = [c for c in subset.columns if c.startswith('dV_dQ_V_')]

        # Convert to numeric vectors
        y_lfp = lfp_sample[feature_cols].astype(float).dropna()
        y_nmc = nmc_sample[feature_cols].astype(float).dropna()

        # x-axis is the voltage this bin discharged INTO, so points read
        # left-to-right as discharge progresses once the axis is inverted
        # below.
        x_lfp = [float(col.split('_')[-2]) for col in y_lfp.index]
        x_nmc = [float(col.split('_')[-2]) for col in y_nmc.index]

        print(f"Plotting {n_steps} steps:")
        print(f"   LFP values: {np.round(y_lfp.values, 3)}")
        print(f"   NMC values: {np.round(y_nmc.values, 3)}")

        ax.plot(x_lfp, y_lfp.values, marker='o', label='LFP', color='#1f77b4', linewidth=2, markersize=6)
        ax.plot(x_nmc, y_nmc.values, marker='s', label='NMC', color='#ff7f0e', linewidth=2, markersize=6)

        ax.set_title(f'GITT Pulse Profile: {n_steps} Steps (Absolute voltage bins)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Terminal Voltage [V]')
        ax.set_ylabel('dV/dQ [V/Ah]')

        all_x = sorted(set(x_lfp + x_nmc), reverse=True)
        if all_x:
            ax.set_xticks(all_x)
        ax.invert_xaxis()  # highest voltage (start of discharge) on the left

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.show()


if __name__ == "__main__":
    input_file = os.path.join("data", "default", "raw", "advanced_synthetic_battery_data.csv")
    step_list = [5, 10, 15, 20, 25]

    datasets = create_features_by_voltage_bins(input_file, step_counts=step_list)
    plot_all_voltage_bin_profiles(datasets)