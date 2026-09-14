"""
Bin-interpolation variant of feature_engineering.py, for the
bin-interpolation-fix experiment in this folder.

Difference from the live feature_engineering.py: that version assigns each
pulse's averaged dV/dQ to only the single 0.1V bin containing the voltage
AFTER the pulse. If a pulse's voltage drop spans multiple bins (which
happens more often at higher discharge current -- see
experiments/04_pulse_variable_discharge_soc_sweep/RESULTS.md's root-cause
section), every bin except the destination one gets nothing for that
battery, even though the voltage clearly passed through it. This version
instead writes the SAME averaged dV/dQ value into every bin between the
pulse's starting and ending voltage, not just the destination bin --
directly patching the "bin gets stepped over between two samples"
mechanism. Everything else (grouping, step-boundary detection, the
>=20% mutual-coverage leakage filter) is unchanged from feature_engineering.py.

This is a feature-extraction-only change -- no simulation protocol or
timing is touched, so it can run against already-generated raw data
without re-simulating anything.
"""
import os
import pandas as pd
import numpy as np


def _default_features_dir(input_csv):
    """Sibling 'features/' dir next to input_csv's 'raw/' dir."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_csv)))
    return os.path.join(run_dir, "features")


def _voltage_to_bin_high(v_val, v_bin_min, v_bin_max, v_bin_width):
    """Round a voltage up to the bin edge it falls under, clamped to the
    pre-initialized bin range -- same convention as the live script."""
    bin_high = np.ceil(round(v_val, 4) * 10) / 10.0
    bin_high = min(v_bin_max, max(v_bin_min + v_bin_width, bin_high))
    return round(bin_high, 1)


def create_features_by_voltage_bins_interpolated(input_csv, step_counts=[5, 10, 15, 20, 25], output_dir=None,
                                                  min_chemistry_coverage=0.2):
    print(f"Loading raw simulation data from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    if output_dir is None:
        output_dir = _default_features_dir(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    groupby_cols = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'N_Steps', 'Variation_ID']
    df['Battery_ID'] = df.groupby(groupby_cols).ngroup()

    all_features = []

    PULSE_DURATION_S = 30 * 60
    REST_DURATION_S = 60 * 60
    STEP_DURATION_S = PULSE_DURATION_S + REST_DURATION_S
    BOUNDARY_TIME_TOL_S = 0.01

    V_BIN_MIN = 1.9
    V_BIN_MAX = 4.3
    V_BIN_WIDTH = 0.1

    for battery_id, group in df.groupby('Battery_ID'):
        group = group.sort_values('Time [s]', kind='stable').reset_index(drop=True)
        n_steps = int(group['N_Steps'].iloc[0])
        times = group['Time [s]'].values

        ocv_rows = []
        for k in range(1, n_steps + 1):
            target_t = k * STEP_DURATION_S
            matches = np.where(np.isclose(times, target_t, atol=BOUNDARY_TIME_TOL_S))[0]
            if matches.size == 0:
                break
            ocv_rows.append(group.iloc[matches[0]])

        if not ocv_rows:
            continue

        ocv_points = pd.DataFrame(ocv_rows).reset_index(drop=True)

        start_row = group.iloc[[0]][['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]
        ocv_sequence = pd.concat([start_row, ocv_points[['Time [s]', 'Voltage [V]', 'Capacity [A.h]']]], ignore_index=True)

        dV = ocv_sequence['Voltage [V]'].diff()
        dQ = ocv_sequence['Capacity [A.h]'].diff()

        valid = (dQ > 1e-5)

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
        for optional_col in ('Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set'):
            if optional_col in group.columns:
                battery_features[optional_col] = group[optional_col].iloc[0]

        n_v_bins = round((V_BIN_MAX - V_BIN_MIN) / V_BIN_WIDTH)
        voltage_bins = [
            f"dV_dQ_V_{round(V_BIN_MIN + i * V_BIN_WIDTH, 1):.1f}_{round(V_BIN_MIN + (i - 1) * V_BIN_WIDTH, 1):.1f}"
            for i in range(n_v_bins, 0, -1)
        ]
        for bin_name in voltage_bins:
            battery_features[bin_name] = np.nan

        # Fill every bin a pulse's voltage span crosses, not just the one
        # containing the ending voltage -- this is the interpolation fix.
        # A small/low-current pulse spans exactly one bin (first==last),
        # so this is a strict superset of the original single-bin logic.
        for idx in dV[valid].index:
            dv_val = dV.loc[idx]
            dq_val = dQ.loc[idx]
            dvdq = dv_val / dq_val

            v_val = ocv_sequence['Voltage [V]'].iloc[idx]
            v_prev = ocv_sequence['Voltage [V]'].iloc[idx - 1]
            v_lo, v_hi = (v_val, v_prev) if v_val <= v_prev else (v_prev, v_val)

            first_bin_high = _voltage_to_bin_high(v_lo, V_BIN_MIN, V_BIN_MAX, V_BIN_WIDTH)
            last_bin_high = _voltage_to_bin_high(v_hi, V_BIN_MIN, V_BIN_MAX, V_BIN_WIDTH)
            n_bins_span = round((last_bin_high - first_bin_high) / V_BIN_WIDTH)

            for b in range(n_bins_span + 1):
                bin_high = round(first_bin_high + b * V_BIN_WIDTH, 1)
                bin_low = round(bin_high - V_BIN_WIDTH, 1)
                bin_key = f"dV_dQ_V_{bin_high:.1f}_{bin_low:.1f}"
                battery_features[bin_key] = dvdq

        all_features.append(battery_features)

    full_df = pd.DataFrame(all_features)

    datasets = {}
    print("\nExtraction Summary:")
    for n in step_counts:
        step_subset = full_df[full_df['N_Steps'] == n].copy()

        all_v_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]
        empty_v_cols = [c for c in all_v_cols if step_subset[c].isna().all()]
        step_subset = step_subset.drop(columns=empty_v_cols)

        feat_cols = [c for c in step_subset.columns if c.startswith('dV_dQ_V_')]

        coverage_by_chem = step_subset.groupby('Chemistry')[feat_cols].apply(lambda g: g.notna().mean())
        one_sided_cols = [c for c in feat_cols if (coverage_by_chem[c] < min_chemistry_coverage).any()]
        step_subset = step_subset.drop(columns=one_sided_cols)
        feat_cols = [c for c in feat_cols if c not in one_sided_cols]

        out_name = os.path.join(output_dir, f"ml_features_{n}_steps.csv")
        step_subset.to_csv(out_name, index=False)
        datasets[n] = step_subset

        valid_vals = step_subset[feat_cols].notna().sum().sum()
        print(f"  -> {out_name}: {len(step_subset)} batteries, {len(feat_cols)} voltage bin columns, {valid_vals} valid entries")

    return datasets
