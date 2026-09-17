"""
Extracts LFP/NMC classification features from raw continuous discharge data.
Computes dV/dQ binned by absolute terminal voltage (0.1V bins) and averaged per bin.
"""
import os
import pandas as pd
import numpy as np


def _default_features_dir(input_csv):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


def create_features_by_voltage_bins(input_csv, output_dir=None, min_chemistry_coverage=0.0):
    print(f"Loading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    df['Battery_ID'] = df['Variation_ID']

    # Expanded bin range: 1.8V to 4.3V covers low-V LFP and high-V NMC ranges
    V_BIN_MIN = 1.8
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

        # Check if Capacity is static/zero
        cap_is_flat = (group['Capacity [A.h]'].nunique() <= 1) or (group['Capacity [A.h]'].abs().max() < 1e-6)

        if cap_is_flat:
            # Reconstruct progress using time delta (in hours) as proxy for Q
            dt = group['Time [s]'].diff().fillna(0).clip(lower=0)
            group['Capacity [A.h]'] = (dt / 3600.0).cumsum()

        dV = group['Voltage [V]'].diff()
        dQ = group['Capacity [A.h]'].diff()

        # Accept valid non-zero steps
        valid = (dQ.abs() > 1e-12) & (dV.abs() > 0)

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
                if abs(dq_val) < 1e-12:
                    continue
                dvdq = dV.loc[idx] / dq_val

                v_val = group['Voltage [V]'].iloc[idx]
                
                # Robust bin mapping
                bin_idx = int(np.floor(round(v_val - V_BIN_MIN, 4) / V_BIN_WIDTH))
                bin_low = round(V_BIN_MIN + bin_idx * V_BIN_WIDTH, 1)
                bin_high = round(bin_low + V_BIN_WIDTH, 1)

                if V_BIN_MIN <= bin_low < V_BIN_MAX:
                    bin_key = f"dV_dQ_V_{bin_high:.1f}_{bin_low:.1f}"
                    bin_values.setdefault(bin_key, []).append(dvdq)

        for bin_name in voltage_bins:
            values = bin_values.get(bin_name)
            battery_features[bin_name] = float(np.mean(values)) if values else np.nan

        all_features.append(battery_features)

    full_df = pd.DataFrame(all_features)

    # Drop columns that are completely NaN across ALL 4 batteries
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
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    input_file = os.path.join(REPO_ROOT, "data", "default", "raw", "calce_data.csv")
    create_features_by_voltage_bins(input_file)