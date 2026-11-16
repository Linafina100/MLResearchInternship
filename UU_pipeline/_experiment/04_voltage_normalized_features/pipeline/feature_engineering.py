"""
experiment 04: voltage-normalized variant of the UU pipeline's feature
engineering. Fork of 03_soc_normalized_features/pipeline/feature_engineering.py
(which normalized capacity -> SOC fraction; see that file for why), adding a
second normalization on the OTHER axis.

Why: even with capacity normalized, bins are still keyed by ABSOLUTE voltage
(1.9-4.3V). Real LFP and real NMC occupy almost disjoint absolute voltage
windows in this dataset (LFP's plateau: ~2.5-3.3V; NMC's pulses: ~3.0-4.1V),
so any "both chemistries must be present in this bin" coverage requirement
-- pooled or per-SOC-bucket -- structurally excludes NMC's only real signal
(the high-voltage region LFP never reaches), because that exclusion is
*exactly* what such a requirement is designed to do to a non-shared region.

Fix: normalize voltage too. Per battery, compute V_min/V_max from its own
observed trace and bin by (V - V_min) / (V_max - V_min) in [0, 1] instead of
absolute volts. Every battery then spans the same normalized range by
construction, regardless of its absolute voltage window, so bins represent
"how far through this battery's own voltage excursion" rather than "what
voltage in volts" -- comparable in principle across chemistries.

Tradeoff (explicit, not hidden): this throws away the absolute-voltage-range
signal entirely (LFP tops out ~3.65V, NMC reaches ~4.2-4.8V -- itself one of
the most legitimate physical LFP-vs-NMC fingerprints there is). What's left
is curve SHAPE only: how dV/d(SOC fraction) varies across the normalized 0-1
voltage-position axis. Chosen deliberately per explicit instruction to drop
absolute voltage rather than keep it as a side feature.

Output columns are named "dV_dSOC_Vfrac_<hi>_<lo>" to make clear neither axis
is in raw physical units anymore.
"""
import os
import numpy as np
import pandas as pd


def _default_features_dir(input_csv):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


def create_features_by_voltage_bins(input_csv, output_dir=None, min_chemistry_coverage=0.2,
                                     exclude_final_transition=True, relative_dsoc_eps=1e-6,
                                     relative_vspan_eps=1e-6):
    print(f"Loading raw data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols, sort=False).ngroup()

    VFRAC_MIN = 0.0
    VFRAC_MAX = 1.0
    VFRAC_WIDTH = 0.1
    n_v_bins = round((VFRAC_MAX - VFRAC_MIN) / VFRAC_WIDTH)
    voltage_bins = [
        f"dV_dSOC_Vfrac_{round(VFRAC_MIN + i * VFRAC_WIDTH, 2)}_{round(VFRAC_MIN + (i - 1) * VFRAC_WIDTH, 2)}"
        for i in range(n_v_bins, 0, -1)
    ]

    print("Sorting and computing vector differences...")
    df = df.sort_values(['Battery_ID', 'Time [s]'], kind='stable').reset_index(drop=True)

    b_id = df['Battery_ID'].values
    v = df['Voltage [V]'].values
    q = df['Capacity [A.h]'].values

    n_batteries = int(b_id.max()) + 1 if len(b_id) else 0

    # Per-battery realized capacity span -> SOC-fraction normalization
    # (unchanged from experiment 03).
    cap_max = np.full(n_batteries, -np.inf)
    cap_min = np.full(n_batteries, np.inf)
    np.maximum.at(cap_max, b_id, q)
    np.minimum.at(cap_min, b_id, q)
    cap_span = cap_max - cap_min

    # Per-battery observed voltage span -> voltage-fraction normalization
    # (new in experiment 04). Dropping absolute voltage entirely means the
    # bin axis below is (V - v_min) / (v_max - v_min) for THIS battery, not
    # volts.
    v_max = np.full(n_batteries, -np.inf)
    v_min = np.full(n_batteries, np.inf)
    np.maximum.at(v_max, b_id, v)
    np.minimum.at(v_min, b_id, v)
    v_span = v_max - v_min
    v_norm = np.divide(v - v_min[b_id], v_span[b_id], out=np.full_like(v, np.nan), where=v_span[b_id] > 0)

    # Valid transitions: within the same battery, a positive SOC-fraction
    # step, and a well-defined (nonzero-span) voltage normalization.
    same_battery = (b_id[1:] == b_id[:-1])
    dV = v[1:] - v[:-1]
    dQ = q[1:] - q[:-1]
    trans_bid_all = b_id[1:]
    span_for_step = cap_span[trans_bid_all]
    dQ_frac = np.divide(dQ, span_for_step, out=np.zeros_like(dQ), where=span_for_step > 0)
    vspan_for_step = v_span[trans_bid_all]

    valid_step = same_battery & (dQ_frac > relative_dsoc_eps) & (vspan_for_step > relative_vspan_eps)

    trans_bid = trans_bid_all[valid_step]
    trans_vnorm = v_norm[1:][valid_step]
    trans_dvdsoc = dV[valid_step] / dQ_frac[valid_step]

    # Exclude PyBaMM cutoff termination artifact (drop last step per battery)
    if exclude_final_transition:
        is_last_step = np.concatenate([trans_bid[:-1] != trans_bid[1:], [True]])
        keep = ~is_last_step
        trans_bid = trans_bid[keep]
        trans_vnorm = trans_vnorm[keep]
        trans_dvdsoc = trans_dvdsoc[keep]

    # Map normalized voltage directly into bin indices
    bin_idx = np.ceil(np.round(trans_vnorm, 6) * 10) / 10.0
    bin_idx = np.clip(bin_idx, VFRAC_MIN + VFRAC_WIDTH, VFRAC_MAX)
    bin_highs = np.round(bin_idx, 2)
    bin_lows = np.round(bin_highs - VFRAC_WIDTH, 2)

    bin_keys = np.array([f"dV_dSOC_Vfrac_{h}_{l}" for h, l in zip(bin_highs, bin_lows)])

    print("Aggregating voltage-fraction bins per battery...")
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
    feat_cols = [c for c in full_df.columns if c.startswith('dV_dSOC_Vfrac_')]
    empty_v_cols = [c for c in feat_cols if full_df[c].isna().all()]
    full_df = full_df.drop(columns=empty_v_cols)
    feat_cols = [c for c in full_df.columns if c.startswith('dV_dSOC_Vfrac_')]

    # Drop bins where no Initial_SOC level has >=20% coverage in EVERY
    # chemistry simultaneously (per-SOC-bucket check, carried over from
    # experiment 03). With voltage normalized, every battery spans the full
    # 0-1 range by construction, so this should no longer structurally
    # exclude either chemistry's bins the way the absolute-voltage version
    # did.
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
    print(f"-> {out_name}: {len(full_df)} batteries, {len(feat_cols)} voltage-fraction bin columns, {valid_vals} valid entries")

    return full_df


if __name__ == "__main__":
    input_file = os.path.join("data", "default", "raw", "synthetic_battery_data.csv")
    create_features_by_voltage_bins(input_file)
