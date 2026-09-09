import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_features_by_steps(input_csv, step_counts=[5, 10, 15, 20, 25]):
    print(f"Loading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    
    # Group each independent simulation run. Variation_ID is included because
    # SOH is stored rounded to 3 decimals, so two distinct random variations
    # (drawn independently per size in simulate_batteries.py) can round to the
    # identical SOH -- without Variation_ID those would silently collide into
    # the same Battery_ID, merging two unrelated runs' overlapping timestamps
    # together and corrupting both their features.
    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'N_Steps', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()
    
    all_features = []
    
    # simulate_batteries.py builds each battery's protocol as N_Steps repeats of
    # ("Discharge ... for 30 minutes", "Rest for 1 hour") -> exactly 5400 s per
    # step, independent of N_Steps (only the pulse current changes). PyBaMM
    # concatenates each named sub-condition's own sub-solution and duplicates
    # the shared boundary timestamp across the join: once as the trailing point
    # of the ending condition, once as the leading point of the next. So the
    # true end-of-relaxation OCV for step k appears at Time == k * STEP_DURATION_S,
    # recorded TWICE — the first occurrence is the still-relaxed voltage, the
    # second already reflects the next pulse's abrupt IR-drop as current
    # switches back on, and must be excluded.
    #
    # Sampling *density* between boundaries varies hugely with cell age/size —
    # a heavily aged cell can be densely sampled throughout its entire pulse
    # and rest, while a fresh cell needs almost no points once settled — so
    # neither "capacity is locally flat" nor "the next time gap is unusually
    # large" reliably distinguishes a genuine step boundary from an ordinary
    # internal sample (both approaches were tried and broke on real batteries).
    # The one thing that's always true, regardless of sampling density, is
    # that a real step boundary sits at an exact multiple of STEP_DURATION_S.
    PULSE_DURATION_S = 30 * 60   # matches PULSE_DURATION in simulate_batteries.py
    REST_DURATION_S = 60 * 60    # matches REST_DURATION in simulate_batteries.py
    STEP_DURATION_S = PULSE_DURATION_S + REST_DURATION_S
    BOUNDARY_TIME_TOL_S = 0.01   # far above float noise (observed to be exact),
                                  # far below the >=0.1 s gap to the next real
                                  # sample after a boundary

    for battery_id, group in df.groupby('Battery_ID'):
        # kind='stable' matters here: at an exact step boundary, the relaxed
        # point and the post-jump point share an identical timestamp, and we
        # rely on their original solve-time order (relaxed first) to tell
        # them apart below.
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
        # Signed on purpose: during discharge voltage drops while capacity rises,
        # so dV/dQ comes out negative, matching the paper's dV/dQ curves (Fig. 3d).
        dV = ocv_sequence['Voltage [V]'].diff()
        dQ = ocv_sequence['Capacity [A.h]'].diff()     # Discharged capacity

        # Keep only valid intervals where capacity actually advanced
        valid = (dQ > 1e-5)

        # Sanity check: aborted pulses (voltage cutoff hit) should only ever
        # truncate the END of the sequence. If an invalid interval is followed
        # by a valid one, enumerate() below would renumber the remaining steps
        # and dV_dQ_step_N would no longer refer to the same physical pulse
        # across batteries, silently breaking positional feature alignment.
        valid_tail = valid.values[1:]  # index 0 is always NaN/False (no prior point)
        invalid_positions = np.where(~valid_tail)[0]
        if invalid_positions.size > 0:
            first_invalid = invalid_positions[0]
            assert valid_tail[first_invalid:].sum() == 0, (
                f"Battery_ID {battery_id}: found a valid dV/dQ transition after an "
                f"invalid one (first invalid at pulse {first_invalid + 1}). Expected "
                "invalid transitions only at the end of the sequence (early voltage "
                "cutoff) — a mid-sequence gap would break positional step indexing."
            )

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

        # full_df is the union of every battery's dV_dQ_step_* columns across
        # ALL step counts, so a subset filtered to N_Steps == n still carries
        # columns beyond n (e.g. dV_dQ_step_6.. for the 5-step subset) that no
        # 5-step battery could ever populate -- entirely NaN. Left in, these
        # get silently dropped by SimpleImputer downstream, which desyncs
        # ml_pipeline_future.py's feature_cols (still listing all of them)
        # from the model's actual trained input width, corrupting the
        # feature-importance plot's column lookup and the saved feature list.
        all_step_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_step_')]
        empty_step_cols = [c for c in all_step_cols if step_subset[c].isna().all()]
        step_subset = step_subset.drop(columns=empty_step_cols)

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
        ax.set_ylabel('dV/dQ [V/Ah]')
        
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