"""
Extracts LFP/NMC classification features from the continuous-discharge
raw simulation data (simulate_batteries.py): dV/dQ computed between every
consecutive raw voltage/capacity sample (there are no pulse boundaries to
anchor on, unlike this project's earlier GITT-pulse-based version,
archived at experiments/08_pulse_protocol_archive/), binned by absolute
terminal voltage (0.1V bins) and averaged per bin -- a dense continuous
trace puts many raw transitions in the same bin, unlike the single
relaxed point the pulse protocol gave it.

Bins only one chemistry ever reaches are dropped (>=20% mutual-coverage
filter): SimpleImputer's median-fill would otherwise give every row of
the missing chemistry an identical constant -- a trivial giveaway, not
real signal. Full writeup: experiments/01_leakage_fix_20_percent_coverage/RESULTS.md.
"""
import os
import pandas as pd
import numpy as np


def _default_features_dir(input_csv):
    """Sibling 'features/' dir next to input_csv's 'raw/' dir, so feature
    CSVs land in the same data/<run_label>/ tree simulate_batteries.py wrote
    the raw data into, sorted by kind alongside it."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


def create_features_by_voltage_bins(input_csv, output_dir=None, min_chemistry_coverage=0.2, v_bin_min=1.9, exclude_final_transition=False):
    print(f"Loading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    # One continuous trace per (chemistry, size, variation) -- already a
    # unique key without a step count.
    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()

    V_BIN_MIN = v_bin_min
    V_BIN_MAX = 4.3
    V_BIN_WIDTH = 0.1
    n_v_bins = round((V_BIN_MAX - V_BIN_MIN) / V_BIN_WIDTH)
    voltage_bins = [
        f"dV_dQ_V_{round(V_BIN_MIN + i * V_BIN_WIDTH, 1):.1f}_{round(V_BIN_MIN + (i - 1) * V_BIN_WIDTH, 1):.1f}"
        for i in range(n_v_bins, 0, -1)
    ]

    all_features = []
    for battery_id, group in df.groupby('Battery_ID'):
        group = group.sort_values('Time [s]', kind='stable').reset_index(drop=True)

        if len(group) < 15:  # skip corrupted/aborted runs
            continue

        # dV/dQ between every consecutive raw sample -- signed (voltage
        # drops while capacity rises during discharge).
        dV = group['Voltage [V]'].diff()
        dQ = group['Capacity [A.h]'].diff()
        valid = (dQ > 1e-5)
        valid_idx = list(dV[valid].index)

        # PyBaMM's "Discharge until X V" is event-triggered, so each
        # battery's final raw sample is always an irregular, oversized
        # last step landing right on the voltage cutoff -- dividing by its
        # small-but-not-negligible dQ produces an exploded, non-physical
        # dV/dQ value (see experiments/11_voltage_cutoff_comparison and
        # experiments/12_exclude_termination_artifact_bins). Dropping just
        # that one transition per battery, rather than whole bins, lets
        # the coverage filter below reject only bins that were *never*
        # genuinely reached by both chemistries.
        if exclude_final_transition and len(valid_idx) > 1:
            valid_idx = valid_idx[:-1]

        if not valid_idx:
            continue

        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Target_Capacity_Ah': group['Target_Capacity_Ah'].iloc[0],
            'Size_Multiplier': group['Size_Multiplier'].iloc[0],
            'SOH': group['SOH'].iloc[0],
            'Initial_SOC': group['Initial_SOC'].iloc[0],
        }
        for optional_col in ('Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set'):
            if optional_col in group.columns:
                battery_features[optional_col] = group[optional_col].iloc[0]

        # Collect every point-to-point dV/dQ value into the bin its ending
        # voltage falls in, then average per bin below.
        bin_values = {}
        for idx in valid_idx:
            dvdq = dV.loc[idx] / dQ.loc[idx]

            v_val = group['Voltage [V]'].iloc[idx]
            bin_high = np.ceil(round(v_val, 4) * 10) / 10.0
            bin_high = min(V_BIN_MAX, max(V_BIN_MIN + V_BIN_WIDTH, bin_high))
            bin_high = round(bin_high, 1)
            bin_low = round(bin_high - V_BIN_WIDTH, 1)

            bin_key = f"dV_dQ_V_{bin_high:.1f}_{bin_low:.1f}"
            bin_values.setdefault(bin_key, []).append(dvdq)

        for bin_name in voltage_bins:
            values = bin_values.get(bin_name)
            battery_features[bin_name] = float(np.mean(values)) if values else np.nan

        all_features.append(battery_features)

    full_df = pd.DataFrame(all_features)

    # Drop bins entirely unreached by anyone.
    all_v_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]
    empty_v_cols = [c for c in all_v_cols if full_df[c].isna().all()]
    full_df = full_df.drop(columns=empty_v_cols)
    feat_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]

    # Drop bins only one chemistry ever reaches (imputation-leakage fix).
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
    input_file = os.path.join("data", "default", "raw", "advanced_synthetic_battery_data.csv")
    create_features_by_voltage_bins(input_file)
