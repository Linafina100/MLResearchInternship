import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_features_by_steps(input_csv, step_counts=[5, 10, 15, 20, 25]):
    print(f"Loading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    
    # Group each independent simulation run
    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'N_Steps']
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()
    
    all_features = []
    
    for battery_id, group in df.groupby('Battery_ID'):
        group = group.sort_values('Time [s]').reset_index(drop=True)
        
        # In GITT discharge, capacity increases during pulse and stays constant during rest
        # Capacity delta between consecutive recorded solver points
        cap_diff = group['Capacity [A.h]'].diff().fillna(0).abs()
        
        # A row is resting if capacity change is virtually zero
        is_resting = cap_diff < 1e-7
        
        # Group contiguous rest blocks
        # state changes whenever we flip between pulse and rest
        state_changes = (is_resting != is_resting.shift(1, fill_value=False)).cumsum()
        
        # Select rows belonging to rest states
        resting_df = group[is_resting]
        if resting_df.empty:
            continue
            
        # Get the VERY LAST row of each rest phase (fully relaxed OCV)
        ocv_points = resting_df.groupby(state_changes).last().reset_index(drop=True)
        
        # Include the initial starting point (t=0) as the reference OCV
        start_row = group.iloc[[0]][['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]
        ocv_sequence = pd.concat([start_row, ocv_points[['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]], ignore_index=True)
        
        # Calculate dV and dQ between consecutive relaxed points
        dV = ocv_sequence['Voltage [V]'].diff().abs() # Drop in voltage
        dQ = ocv_sequence['Capacity [A.h]'].diff()     # Discharged capacity
        
        # Keep only valid intervals where capacity actually advanced
        valid = (dQ > 1e-5)
        dvdq_values = (dV[valid] / dQ[valid]).values
        
        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Target_Capacity_Ah': group['Target_Capacity_Ah'].iloc[0],
            'Size_Multiplier': group['Size_Multiplier'].iloc[0],
            'SOH': group['SOH'].iloc[0],
            'Initial_SOC': group['Initial_SOC'].iloc[0],
            'N_Steps': int(group['N_Steps'].iloc[0]),
        }
        
        for step_idx, val in enumerate(dvdq_values):
            battery_features[f'dV_dQ_step_{step_idx+1}'] = val
            
        all_features.append(battery_features)
        
    full_df = pd.DataFrame(all_features)
    
    # Split into 5 datasets
    datasets = {}
    print("\nExtraction Summary:")
    for n in step_counts:
        step_subset = full_df[full_df['N_Steps'] == n].copy()
        
        # Find all valid step columns for this step count
        feat_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_step_')]
        
        out_name = f"ml_features_{n}_steps.csv"
        step_subset.to_csv(out_name, index=False)
        datasets[n] = step_subset
        
        # Sanity check: count non-null values
        valid_vals = step_subset[feat_cols].notna().sum().sum()
        print(f"  -> {out_name}: {len(step_subset)} batteries, {len(feat_cols)} step columns, {valid_vals} non-null values")
        
    return datasets


def plot_all_step_profiles(datasets):
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
        feature_cols = [c for c in subset.columns if c.startswith('dV_dQ_step_')]
        
        # Convert to numeric vectors
        y_lfp = lfp_sample[feature_cols].astype(float).dropna()
        y_nmc = nmc_sample[feature_cols].astype(float).dropna()
        
        x_lfp = [int(col.split('_')[-1]) for col in y_lfp.index]
        x_nmc = [int(col.split('_')[-1]) for col in y_nmc.index]
        
        print(f"Plotting {n_steps} steps:")
        print(f"   LFP values: {np.round(y_lfp.values, 3)}")
        print(f"   NMC values: {np.round(y_nmc.values, 3)}")
        
        ax.plot(x_lfp, y_lfp.values, marker='o', label='LFP', color='#1f77b4', linewidth=2, markersize=6)
        ax.plot(x_nmc, y_nmc.values, marker='s', label='NMC', color='#ff7f0e', linewidth=2, markersize=6)
        
        ax.set_title(f'GITT Pulse Profile: {n_steps} Steps (ΔQ = {0.6 / n_steps:.3f} Ah / pulse)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Pulse Step Number')
        ax.set_ylabel('|dV/dQ| [V/Ah]')
        
        all_x = sorted(list(set(x_lfp + x_nmc)))
        if all_x:
            ax.set_xticks(all_x)
            
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    #plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_file = "advanced_synthetic_battery_data.csv"
    step_list = [5, 10, 15, 20, 25]
    
    datasets = create_features_by_steps(input_file, step_counts=step_list)
    plot_all_step_profiles(datasets)