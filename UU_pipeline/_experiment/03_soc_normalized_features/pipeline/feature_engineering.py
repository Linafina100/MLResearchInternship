"""
experiment 03: SOC-normalized variant of the UU pipeline's feature
engineering. Fork of 02_nmc_contiguous_segment_fix/pipeline/feature_engineering.py
(itself unchanged from experiment 01) that replaces dV/dQ (absolute Ah) with
dV/d(SOC fraction) per transition.

Why: with the NMC segmentation bug fixed (experiment 02), every real battery's
dV/dQ feature came out on a wildly different absolute scale than synthetic
batteries -- real NMC segments max out around 0.000113 Ah total (vs. the
~2 Ah synthetic cells), and real LFP segments sit around 0.0014 Ah. Dividing
a comparable dV by a dQ that's 1000-20000x smaller inflates dV/dQ by the same
factor, so real cells land far outside the synthetic training distribution on
a purely unit-scale basis -- not because of any real electrochemical
difference. None of the real capacity values were ever rescaled to the
synthetic ~2 Ah target (the Target_Capacity_Ah field is stored as metadata
but never applied).

Fix: normalize dQ by each battery's OWN realized capacity span before taking
the derivative, i.e. use dV/d(SOC fraction) where SOC fraction = (Q - Q_min) /
(Q_max - Q_min) for that specific battery. This makes the resulting feature
scale-invariant to the battery's absolute capacity, so a tiny real segment
and a full-size synthetic cell become directly comparable. Output columns
are named "dV_dSOC_V_<hi>_<lo>" (not "dV_dQ_V_...") to make the changed units
explicit to downstream code.
"""
import os
import numpy as np
import pandas as pd


def _default_features_dir(input_csv):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


def create_features_by_voltage_bins(input_csv, output_dir=None, min_chemistry_coverage=0.2, v_bin_min=1.9,
                                     exclude_final_transition=True, relative_dsoc_eps=1e-6):
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
        f"dV_dSOC_V_{round(V_BIN_MIN + i * V_BIN_WIDTH, 1):.1f}_{round(V_BIN_MIN + (i - 1) * V_BIN_WIDTH, 1):.1f}"
        for i in range(n_v_bins, 0, -1)
    ]

    print("Sorting and computing vector differences...")
    df = df.sort_values(['Battery_ID', 'Time [s]'], kind='stable').reset_index(drop=True)

    b_id = df['Battery_ID'].values
    v = df['Voltage [V]'].values
    q = df['Capacity [A.h]'].values

    # Per-battery realized capacity span (Q_max - Q_min), used to convert
    # each battery's own dQ into a SOC fraction regardless of its absolute
    # capacity scale.
    n_batteries = int(b_id.max()) + 1 if len(b_id) else 0
    cap_span = np.zeros(n_batteries)
    cap_max = np.full(n_batteries, -np.inf)
    cap_min = np.full(n_batteries, np.inf)
    np.maximum.at(cap_max, b_id, q)
    np.minimum.at(cap_min, b_id, q)
    cap_span = cap_max - cap_min

    # Valid transitions: within the same battery and a positive SOC-fraction step
    same_battery = (b_id[1:] == b_id[:-1])
    dV = v[1:] - v[:-1]
    dQ = q[1:] - q[:-1]
    trans_bid_all = b_id[1:]
    span_for_step = cap_span[trans_bid_all]
    dQ_frac = np.divide(dQ, span_for_step, out=np.zeros_like(dQ), where=span_for_step > 0)

    valid_step = same_battery & (dQ_frac > relative_dsoc_eps)

    trans_bid = trans_bid_all[valid_step]
    trans_v = v[1:][valid_step]
    trans_dvdsoc = dV[valid_step] / dQ_frac[valid_step]

    # Exclude PyBaMM cutoff termination artifact (drop last step per battery)
    if exclude_final_transition:
        is_last_step = np.concatenate([trans_bid[:-1] != trans_bid[1:], [True]])
        keep = ~is_last_step
        trans_bid = trans_bid[keep]
        trans_v = trans_v[keep]
        trans_dvdsoc = trans_dvdsoc[keep]

    # Map voltages directly into bin indices via math instead of string loops
    bin_idx = np.ceil(np.round(trans_v, 4) * 10) / 10.0
    bin_idx = np.clip(bin_idx, V_BIN_MIN + V_BIN_WIDTH, V_BIN_MAX)
    bin_highs = np.round(bin_idx, 1)
    bin_lows = np.round(bin_highs - V_BIN_WIDTH, 1)

    bin_keys = np.array([f"dV_dSOC_V_{h:.1f}_{l:.1f}" for h, l in zip(bin_highs, bin_lows)])

    print("Aggregating voltage bins per battery...")
    steps_df = pd.DataFrame({
        'Battery_ID': trans_bid,
        'Bin': bin_keys,
        'dV_dSOC': trans_dvdsoc
    })

    features_pivot = steps_df.groupby(['Battery_ID', 'Bin'])['dV_dSOC'].mean().unstack()

    existing_bins = [b for b in voltage_bins if b in features_pivot.columns]
    features_pivot = features_pivot.reindex(columns=existing_bins)

    meta_cols = ['Battery_ID', 'Chemistry', 'Target_Capacity_Ah', 'Size_Multiplier', 'SOH', 'Initial_SOC']
    for opt in ('C_Rate', 'Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set'):
        if opt in df.columns:
            meta_cols.append(opt)

    meta_df = df.drop_duplicates('Battery_ID')[meta_cols].set_index('Battery_ID')

    full_df = meta_df.join(features_pivot, how='inner').reset_index()

    # Filter out abortive/too short batteries (<15 raw points)
    pt_counts = df.groupby('Battery_ID').size()
    valid_bids = pt_counts[pt_counts >= 15].index
    full_df = full_df[full_df['Battery_ID'].isin(valid_bids)].copy()

    # Drop bins entirely unreached
    feat_cols = [c for c in full_df.columns if c.startswith('dV_dSOC_V_')]
    empty_v_cols = [c for c in feat_cols if full_df[c].isna().all()]
    full_df = full_df.drop(columns=empty_v_cols)
    feat_cols = [c for c in full_df.columns if c.startswith('dV_dSOC_V_')]

    # Drop bins where no Initial_SOC level has >=20% coverage in EVERY
    # chemistry simultaneously. Pooling coverage across all SOC levels (as
    # experiment 01/02 did) dropped every bin here: LFP's plateau (2.6-3.3V)
    # and NMC's sloped curve (3.4-4.1V) barely overlap, and each real battery
    # is deliberately truncated to a narrow tail-end voltage window (by
    # design of the SOC sweep), so per-bin coverage stays low when diluted
    # across all 12 truncation depths. Checking per-SOC-level instead asks
    # the right question: at THIS truncation depth, do both chemistries
    # actually reach this bin.
    chemistries = full_df['Chemistry'].unique()
    coverage_by_soc_chem = full_df.groupby(['Initial_SOC', 'Chemistry'])[feat_cols].apply(lambda g: g.notna().mean())

    def bin_survives(col):
        for soc_val, grp in coverage_by_soc_chem[col].groupby(level=0):
            grp = grp.droplevel(0)
            if len(grp) == len(chemistries) and (grp >= min_chemistry_coverage).all():
                return True
        return False

    one_sided_cols = [c for c in feat_cols if not bin_survives(c)]
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
