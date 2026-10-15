"""
Extracts LFP/NMC classification features from continuous-discharge raw data.
Vectorized implementation replacing slow Python loops with NumPy vector arithmetic.
"""
import os
import numpy as np
import pandas as pd


def _default_features_dir(input_csv):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


def create_features_by_voltage_bins(input_csv, output_dir=None, min_chemistry_coverage=0.2, v_bin_min=1.9, exclude_final_transition=True):
    print(f"Loading raw data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols, sort=False).ngroup()

    V_BIN_MIN = v_bin_min
    V_BIN_MAX = 4.3
    V_BIN_WIDTH = 0.1
    n_v_bins = round((V_BIN_MAX - V_BIN_MIN) / V_BIN_WIDTH)
    voltage_bins = [
        f"dV_dQ_V_{round(V_BIN_MIN + i * V_BIN_WIDTH, 1):.1f}_{round(V_BIN_MIN + (i - 1) * V_BIN_WIDTH, 1):.1f}"
        for i in range(n_v_bins, 0, -1)
    ]

    print("Sorting and computing vector differences...")
    # Sort stably so consecutive sample differences are accurate
    df = df.sort_values(['Battery_ID', 'Time [s]'], kind='stable').reset_index(drop=True)

    '''
    CHANGES MADE FOR EVALUATE TO BE FASTER
     The script previuosly grouped the entire dataset by Battery_ID
    and iterated through every single battery using a for loop. 
    Inside that loop, it iterated over every measurementsample using
    .loc[] and iloc.[] to comnpute the scalar differences.
    
    for battery_id, group in df.groupby('Battery_ID'):
    for idx in valid_idx:
        dvdq = dV.loc[idx] / dQ.loc[idx]
    
    Running millsions of dicts and scalar slices created a severre bottleneck. 
    The new implementation uses vectorized NumPy operations to compute the differences in one go, 
    which is significantly faster.

    NEW APPROACH:
    The entire table is sorted once by Batter_ID anf Time. 
    Consecutive differences are computed across the entire underlying NumPy arrays in a single vectorized operation.
    This delegades the gradient computation directly to NumPy, which is implemented in C and optimized for performance.
    '''

    
    # Vectorized differences
    b_id = df['Battery_ID'].values
    v = df['Voltage [V]'].values
    q = df['Capacity [A.h]'].values

    # Valid transitions: within the same battery and positive dQ
    same_battery = (b_id[1:] == b_id[:-1])
    dV = v[1:] - v[:-1]
    dQ = q[1:] - q[:-1]
    valid_step = same_battery & (dQ > 1e-5)

    # Transition endpoints
    trans_bid = b_id[1:][valid_step]
    trans_v = v[1:][valid_step]
    trans_dvdq = dV[valid_step] / dQ[valid_step]

    # Exclude PyBaMM cutoff termination artifact (drop last step per battery)
    '''preserved that the last step of each battery is dropped, as it may be an artifact of the PyBaMM simulation cutoff.
    New implementation uses a boolean mask to identify the last step of each battery and exclude it from the transition arrays
    instead of iterating through each battery and checking the last index. This is more efficient and avoids Python loops.'''

    if exclude_final_transition:
        is_last_step = np.concatenate([trans_bid[:-1] != trans_bid[1:], [True]])
        keep = ~is_last_step
        trans_bid = trans_bid[keep]
        trans_v = trans_v[keep]
        trans_dvdq = trans_dvdq[keep]

    '''PREVIOUS APPROACH:
    Every sample evaluated np.ceil(round(v_val, 4) * 10) / 10.0, clamped bounds, formatted an f string,
    and append the value to an individual dict lsit. 
    This was slow because it iterated over every sample in Python and created many temporary objects.
    
    NEW APPROACH: 
    Bin endpoints are assigned simultaneously across all valid transitions using NumPy mathematical operations,
    and zipped once to generate bin labels across the whole array.
    This avoids the overhead of Python loops and temporary objects, and is much faster.
    '''
    # Map voltages directly into bin indices via math instead of string loops
    bin_idx = np.ceil(np.round(trans_v, 4) * 10) / 10.0
    bin_idx = np.clip(bin_idx, V_BIN_MIN + V_BIN_WIDTH, V_BIN_MAX)
    bin_highs = np.round(bin_idx, 1)
    bin_lows = np.round(bin_highs - V_BIN_WIDTH, 1)

    bin_keys = np.array([f"dV_dQ_V_{h:.1f}_{l:.1f}" for h, l in zip(bin_highs, bin_lows)])

    print("Aggregating voltage bins per battery...")
    # Create DataFrame of all individual steps and compute the mean per (Battery_ID, bin)
    '''PREVIOUS APPROACH:
    Iterated over voltage_bins for every battery and called float(np.mean(values)) indidually
    before converting a list of dicts to a dataframe. 
    NEW APPROACH:
    uses an aggregated group by operation followed by an unstack into feature matrix. 
    The resulting table is then joined back to deduplicated battery metadata columns, in a single indexed operation
    '''
    
    steps_df = pd.DataFrame({
        'Battery_ID': trans_bid,
        'Bin': bin_keys,
        'dV_dQ': trans_dvdq
    })
    
    # Fast pivot to get battery rows and bin columns
    features_pivot = steps_df.groupby(['Battery_ID', 'Bin'])['dV_dQ'].mean().unstack()

    # Reindex columns to preserve standardized descending order
    existing_bins = [b for b in voltage_bins if b in features_pivot.columns]
    features_pivot = features_pivot.reindex(columns=existing_bins)

    # Extract metadata per battery rapidly
    meta_cols = ['Battery_ID', 'Chemistry', 'Target_Capacity_Ah', 'Size_Multiplier', 'SOH', 'Initial_SOC']
    for opt in ('C_Rate', 'Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set'):
        if opt in df.columns:
            meta_cols.append(opt)

    meta_df = df.drop_duplicates('Battery_ID')[meta_cols].set_index('Battery_ID')
    
    # Merge metadata with pivoted bins
    full_df = meta_df.join(features_pivot, how='inner').reset_index()

    # Filter out abortive/too short batteries (<15 raw points)
    pt_counts = df.groupby('Battery_ID').size()
    valid_bids = pt_counts[pt_counts >= 15].index
    full_df = full_df[full_df['Battery_ID'].isin(valid_bids)].copy()

    # Drop bins entirely unreached
    feat_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]
    empty_v_cols = [c for c in feat_cols if full_df[c].isna().all()]
    full_df = full_df.drop(columns=empty_v_cols)
    feat_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]

    # Drop bins where one chemistry has <20% coverage
    coverage_by_chem = full_df.groupby('Chemistry')[feat_cols].apply(lambda g: g.notna().mean())
    one_sided_cols = [c for c in feat_cols if (coverage_by_chem[c] < min_chemistry_coverage).any()]
    full_df = full_df.drop(columns=one_sided_cols)
    feat_cols = [c for c in feat_cols if c not in one_sided_cols]

    out_name = os.path.join(output_dir, "ml_features.csv")
    full_df.to_csv(out_name, index=False)

    valid_vals = full_df[feat_cols].notna().sum().sum()
    print(f"-> {out_name}: {len(full_df)} batteries, {len(feat_cols)} voltage bin columns, {valid_vals} valid entries")

    return full_df


if __name__ == "__main__":
    input_file = os.path.join("data", "default", "raw", "synthetic_battery_data.csv")
    create_features_by_voltage_bins(input_file)