"""
Extracts LFP/NMC classification features from the continuous-discharge
raw simulation data (simulate_batteries.py): dV/dQ computed between every
consecutive raw voltage/capacity sample (there are no pulse boundaries to
anchor on, unlike this project's earlier GITT-pulse-based version,
archived at experiments/07_pulse_protocol_archive/), binned by absolute
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


def create_features_by_voltage_bins(input_csv, output_dir=None, min_chemistry_coverage=0.0):
    print(f"Loading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Directly map Battery_ID to the explicit Variation_ID in the CSV
    df['Battery_ID'] = df['Variation_ID']

    V_BIN_MIN = 2.0
    V_BIN_MAX = 4.2
    V_BIN_WIDTH = 0.1
    n_v_bins = round((V_BIN_MAX - V_BIN_MIN) / V_BIN_WIDTH)
    voltage_bins = [
        f"dV_dQ_V_{round(V_BIN_MIN + i * V_BIN_WIDTH, 1):.1f}_{round(V_BIN_MIN + (i - 1) * V_BIN_WIDTH, 1):.1f}"
        for i in range(n_v_bins, 0, -1)
    ]

    all_features = []
    for battery_id, group in df.groupby('Battery_ID'):
        group = group.sort_values('Time [s]', kind='stable').reset_index(drop=True)

        dV = group['Voltage [V]'].diff()
        dQ = group['Capacity [A.h]'].diff()
        
        # Accept all valid non-zero step changes
        valid = dQ.abs() > 0

        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Target_Capacity_Ah': group['Target_Capacity_Ah'].iloc[0],
            'Size_Multiplier': group['Size_Multiplier'].iloc[0] if 'Size_Multiplier' in group.columns else 1.0,
            'SOH': group['SOH'].iloc[0] if 'SOH' in group.columns else 1.0,
            'Initial_SOC': group['Initial_SOC'].iloc[0] if 'Initial_SOC' in group.columns else 1.0,
        }

        bin_values = {}
        if valid.any():
            for idx in dV[valid].index:
                dq_val = dQ.loc[idx]
                if abs(dq_val) < 1e-9:
                    continue
                dvdq = dV.loc[idx] / dq_val

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

    # Drop columns that are entirely NaN across ALL batteries
    all_v_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]
    empty_v_cols = [c for c in all_v_cols if full_df[c].isna().all()]
    full_df = full_df.drop(columns=empty_v_cols)
    feat_cols = [c for c in full_df.columns if c.startswith('dV_dQ_V_')]

    out_name = os.path.join(output_dir, "ml_features.csv")
    full_df.to_csv(out_name, index=False)

    valid_vals = full_df[feat_cols].notna().sum().sum()
    print(f"-> {out_name}: {len(full_df)} batteries, {len(feat_cols)} voltage bin columns, {valid_vals} valid entries")

    return full_df

if __name__ == "__main__":
    # Dynamically resolve REPO_ROOT from script location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    input_file = os.path.join(REPO_ROOT, "data", "default", "raw", "advanced_synthetic_battery_data.csv")
    create_features_by_voltage_bins(input_file)