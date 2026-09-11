import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    # Absolute terminal-voltage bins, not SOC bins. A real used cell pulled
    # off the line has no known SOC without already knowing its chemistry
    # (and remaining capacity) -- that's the target variable, so keying
    # features on computed SOC is unusable at inference time. Voltage is
    # exactly what a GITT pulse test measures directly, with no dependency
    # on Target_Capacity_Ah/Initial_SOC (both simulation-only ground truth
    # that a real cell doesn't have). The range is a superset of both
    # chemistries' plausible terminal-voltage span (LFP ~2.0-3.6 V, NMC
    # ~2.5-4.2 V per Chen2020/Prada2013) with margin; bins outside what a
    # chemistry ever actually reaches are dropped per step-count below, the
    # same way unreached SOC bins were dropped before.
    V_BIN_MIN = 1.9
    V_BIN_MAX = 4.3
    V_BIN_WIDTH = 0.1

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

        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Target_Capacity_Ah': group['Target_Capacity_Ah'].iloc[0],
            'Size_Multiplier': group['Size_Multiplier'].iloc[0],
            'SOH': group['SOH'].iloc[0],
            'Initial_SOC': group['Initial_SOC'].iloc[0],
            'N_Steps': int(group['N_Steps'].iloc[0]),
        }
        # Carried through for analysis/debugging only (e.g. checking whether
        # the voltage-bin separation holds up per base parameter set or
        # temperature); ml_pipeline_future.py's metadata_cols excludes these
        # from the actual feature matrix.
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

        # Map each valid dV/dQ transition to the absolute terminal-voltage
        # range it ends in. Phase 1's dV_dQ_step_N was positional (which
        # pulse number), which real-world truncated discharge data can't
        # reproduce (a partial factory pull doesn't know it was "pulse 7 of
        # 15"). Phase 2 anchored bins to computed SOC instead, but a real
        # used cell's SOC can't be computed without already knowing its
        # chemistry and true capacity -- both the target variable and an
        # unmeasured quantity for an unidentified cell. Voltage is what the
        # pulse test actually measures, directly, with no such dependency.
        for idx in dV[valid].index:
            dv_val = dV.loc[idx]
            dq_val = dQ.loc[idx]
            dvdq = dv_val / dq_val

            # Terminal voltage at this OCV point (the value this transition
            # discharged INTO), clamped to the pre-initialized bin range.
            v_val = ocv_sequence['Voltage [V]'].iloc[idx]
            bin_high = np.ceil(round(v_val, 4) * 10) / 10.0
            bin_high = min(V_BIN_MAX, max(V_BIN_MIN + V_BIN_WIDTH, bin_high))
            bin_high = round(bin_high, 1)
            bin_low = round(bin_high - V_BIN_WIDTH, 1)

            bin_key = f"dV_dQ_V_{bin_high:.1f}_{bin_low:.1f}"
            battery_features[bin_key] = dvdq

        all_features.append(battery_features)
        
    full_df = pd.DataFrame(all_features)
    
    # Split into 5 datasets
    datasets = {}
    print("\nExtraction Summary:")
    for n in step_counts:
        step_subset = full_df[full_df['N_Steps'] == n].copy()

        # All voltage-bin columns are pre-initialized for every battery
        # regardless of N_Steps, so a bin is only ever entirely NaN for a
        # given step count if that protocol's coarser per-pulse capacity
        # (or a chemistry's voltage range) never actually lands a transition
        # inside it -- drop those so ml_pipeline.py's feature_cols stays in
        # sync with the model's real trained input width (see the analogous
        # issue this fixed for the old positional dV_dQ_step_* columns).
        all_v_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]
        empty_v_cols = [c for c in all_v_cols if step_subset[c].isna().all()]
        step_subset = step_subset.drop(columns=empty_v_cols)

        # Find all valid voltage bin columns for this step count
        feat_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]

        # Drop bins that only ONE chemistry ever reaches. LFP and NMC have
        # different voltage ranges (LFP ~2.0-3.6V, NMC ~2.5-4.2V per
        # Prada2013/Chen2020), so a bin near either ceiling can be ~0%
        # populated for one chemistry while well-populated for the other.
        # ml_pipeline.py's SimpleImputer(strategy='median') then fills every
        # one of that chemistry's rows in the bin with an identical constant
        # derived from the other chemistry's real values -- a trivial
        # "does this feature equal that exact constant?" split lets a tree
        # use the bin as a disguised chemistry indicator instead of learning
        # real dV/dQ shape, inflating accuracy without genuine signal.
        # Requiring both chemistries to have real (non-imputed) values at
        # least min_chemistry_coverage of the time keeps only bins where
        # both chemistries contribute genuine, varying measurements.
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

        # x-axis is the voltage this bin discharged INTO (the "_high"
        # boundary, e.g. "dV_dQ_V_3.8_3.7" -> 3.8), so points read
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

    #plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_file = os.path.join("data", "default", "raw", "advanced_synthetic_battery_data.csv")
    step_list = [5, 10, 15, 20, 25]

    datasets = create_features_by_voltage_bins(input_file, step_counts=step_list)
    plot_all_voltage_bin_profiles(datasets)