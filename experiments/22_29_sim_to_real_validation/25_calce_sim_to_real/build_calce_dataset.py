"""
Experiment 25: parses the 4 real CALCE cell files
(data/calce_experiment_data/, 2 LFP + 2 NMC single discharge traces) into
the same (Time [s], Voltage [V], Capacity [A.h], Chemistry, Variation_ID, ...)
shape root feature_engineering.py expects -- same output shape
build_real_empa_dataset.py produces, so this reuses root
feature_engineering.py/ml_pipeline.py directly rather than forking its own
copy the way experiments/09_calce_data_integration/ did.

This is a fresh copy of experiments/09_calce_data_integration's own
parse_calce_discharge_file/merge_calce_dataset, NOT an edit of that
experiment's files -- experiment 09 belongs to a colleague and a prior
commit (`git show 3328d42`, branch `verify-experiment-09-locally`, never
pushed/merged "per instruction") explicitly fixed the bug below locally
without touching/merging into that experiment, so this one carries the
same one-hunk fix forward into its own copy instead:

Bug (still present in experiments/09_calce_data_integration/
build_calce_dataset.py today): `'cap' in c_lower` matched BOTH
`Charge_Capacity(Ah)` and `Discharge_Capacity(Ah)` column headers, and the
duplicate-column dedup that followed silently kept whichever came first in
the sheet -- the always-zero charge-capacity column for 3 of the 4 CALCE
files, losing all their capacity data. Fixed here by matching
discharge-capacity terms first and explicitly skipping charge-capacity
columns.
"""
import os
import glob
import pandas as pd
import numpy as np


def parse_calce_discharge_file(file_path, chemistry, target_capacity_ah, variation_id):
    """Parses a single CALCE Excel/CSV file, handling metadata header rows,
    unit conversions (mV to V), and column mappings safely."""
    print(f"Processing ({chemistry}): {os.path.basename(file_path)}...")

    df_raw = None
    if file_path.endswith(('.xlsx', '.xls')):
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        target_sheet = sheet_names[0]

        for sheet in sheet_names:
            if any(term in str(sheet).lower() for term in ['data', 'channel', 'sheet1']):
                target_sheet = sheet
                break

        preview = pd.read_excel(file_path, sheet_name=target_sheet, nrows=30, header=None)
        header_row = 0

        for idx, row in preview.iterrows():
            row_str = " ".join([str(val).lower() for val in row.dropna().values])
            if any(term in row_str for term in ['volt', 'ecell', 'mv', 'time', 'duration', 'step', 'current']):
                header_row = idx
                break

        df_raw = pd.read_excel(file_path, sheet_name=target_sheet, skiprows=header_row)
    else:
        df_raw = pd.read_csv(file_path)

    df_raw.columns = [str(col).strip() for col in df_raw.columns]

    col_map = {}
    for col in df_raw.columns:
        c_lower = str(col).lower()
        if 'time' in c_lower or 'duration' in c_lower:
            col_map[col] = 'Time [s]'
        elif 'volt' in c_lower or 'ecell' in c_lower or 'mv' in c_lower or c_lower == 'v':
            col_map[col] = 'Voltage [V]'
        elif 'curr' in c_lower or 'ma' in c_lower or 'i/a' in c_lower or c_lower == 'a':
            col_map[col] = 'Current [A]'
        elif 'discharge' in c_lower and ('cap' in c_lower or 'q_dis' in c_lower or 'mah' in c_lower):
            # Fix: match discharge-capacity explicitly first so it always
            # wins over the charge-capacity column below (see docstring).
            col_map[col] = 'Capacity [A.h]'
        elif 'charge' in c_lower and 'cap' in c_lower:
            continue  # charge-only capacity -- skip, do not collide with discharge capacity
        elif 'cap' in c_lower or 'mah' in c_lower:
            col_map[col] = 'Capacity [A.h]'

    df = df_raw.rename(columns=col_map).copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    def get_numeric_series(dataframe, column_name):
        if column_name in dataframe.columns:
            data = dataframe[column_name]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            return pd.to_numeric(data, errors='coerce')
        return None

    volt_numeric = get_numeric_series(df, 'Voltage [V]')
    if volt_numeric is not None:
        df['Voltage [V]'] = volt_numeric / 1000.0 if volt_numeric.abs().max() > 100 else volt_numeric

    curr_numeric = get_numeric_series(df, 'Current [A]')
    if curr_numeric is not None:
        df['Current [A]'] = curr_numeric / 1000.0 if curr_numeric.abs().max() > 100 else curr_numeric

    cap_numeric = get_numeric_series(df, 'Capacity [A.h]')
    if cap_numeric is not None:
        df['Capacity [A.h]'] = cap_numeric / 1000.0 if cap_numeric.abs().max() > 100 else cap_numeric

    if 'Time [s]' not in df.columns or 'Voltage [V]' not in df.columns:
        raise ValueError(f"Could not automatically detect Time/Voltage columns. Found columns: {list(df_raw.columns)[:5]}")

    time_numeric = get_numeric_series(df, 'Time [s]')
    df['Time [s]'] = time_numeric
    df = df.dropna(subset=['Time [s]', 'Voltage [V]'])

    if 'Current [A]' in df.columns and not df['Current [A]'].isna().all():
        discharge_df = df[df['Current [A]'] < -1e-4].copy()
    else:
        discharge_df = df[df['Voltage [V]'].diff() <= 0].copy()

    if discharge_df.empty:
        discharge_df = df.copy()

    discharge_df = discharge_df.sort_values('Time [s]').reset_index(drop=True)

    if 'Capacity [A.h]' not in discharge_df.columns or discharge_df['Capacity [A.h]'].isna().all():
        dt = discharge_df['Time [s]'].diff().fillna(0)
        current_a = discharge_df['Current [A]'].abs() if 'Current [A]' in discharge_df.columns else 0.05 * target_capacity_ah
        discharge_df['Capacity [A.h]'] = (current_a * dt).cumsum() / 3600.0
    else:
        discharge_df['Capacity [A.h]'] = discharge_df['Capacity [A.h]'] - discharge_df['Capacity [A.h]'].iloc[0]

    actual_q_discharged = discharge_df['Capacity [A.h]'].iloc[-1]
    soh_val = round(float(actual_q_discharged / target_capacity_ah), 4)
    soh_val = min(1.0, max(0.5, soh_val))

    discharge_df['Chemistry'] = chemistry
    discharge_df['Target_Capacity_Ah'] = target_capacity_ah
    discharge_df['Size_Multiplier'] = 1.0
    discharge_df['SOH'] = soh_val
    discharge_df['Initial_SOC'] = 1.0
    discharge_df['Variation_ID'] = variation_id
    discharge_df['Ambient_Temperature_C'] = 25.0
    discharge_df['Source_File'] = os.path.basename(file_path)

    required_cols = [
        'Time [s]', 'Voltage [V]', 'Capacity [A.h]',
        'Chemistry', 'Target_Capacity_Ah', 'Size_Multiplier',
        'SOH', 'Initial_SOC', 'Variation_ID', 'Ambient_Temperature_C', 'Source_File'
    ]
    return discharge_df[required_cols]


def merge_calce_dataset(raw_input_dir, output_csv):
    all_traces = []

    dataset_configs = [
        {"pattern": "*A1-*", "chem": "LFP", "nominal_ah": 1.1},
        {"pattern": "*SP20*", "chem": "NMC", "nominal_ah": 2.0},
    ]

    variation_counter = 1
    processed_files = set()

    for config in dataset_configs:
        matching_files = sorted(glob.glob(os.path.join(raw_input_dir, "**", config["pattern"]), recursive=True))
        valid_files = [f for f in matching_files if f.endswith(('.xlsx', '.xls', '.csv'))]

        for fpath in valid_files:
            if fpath in processed_files:
                continue
            trace_df = parse_calce_discharge_file(
                file_path=fpath, chemistry=config["chem"],
                target_capacity_ah=config["nominal_ah"], variation_id=variation_counter,
            )
            all_traces.append(trace_df)
            processed_files.add(fpath)
            variation_counter += 1

    combined_df = pd.concat(all_traces, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"\n--> Saved merged CALCE raw dataset to: {output_csv}")
    print(f"Total rows: {len(combined_df)} | Total unique battery traces: {combined_df['Variation_ID'].nunique()}")
    print(combined_df.groupby(['Variation_ID', 'Chemistry', 'Source_File']).agg(
        n_rows=('Time [s]', 'size'),
        max_capacity_ah=('Capacity [A.h]', 'max'),
        soh=('SOH', 'first'),
        v_min=('Voltage [V]', 'min'),
        v_max=('Voltage [V]', 'max'),
    ).reset_index().to_string(index=False))
    return combined_df


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    CALCE_DIR = os.path.join(REPO_ROOT, "data", "calce_experiment_data")
    OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "25_calce_real", "raw", "calce_raw.csv")

    merge_calce_dataset(CALCE_DIR, OUTPUT_FILE)
