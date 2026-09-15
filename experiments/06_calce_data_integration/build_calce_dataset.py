"""
This script automates the process of finding raw CALCE excel/csv files
in the data/calce_experiment_data/ folder, extracting the continuous discharge
phases, normalizing column names and units, calculating capacity and SOH, and
writing a unified dataset to data/default/raw/advanced_synthetic_battery_data.csv.
"""

### IMPORTS
import os
import glob
import pandas as pd
import numpy as np

### FUNCTION THAT PROCESSES ONE CALCE FILE
def parse_calce_discharge_file(file_path, chemistry, target_capacity_ah, variation_id):
    """
    Parses a single CALCE Excel/CSV file, handling metadata header rows,
    unit conversions (mV to V), and column mappings.
    """
    print(f"Processing ({chemistry}): {os.path.basename(file_path)}...")
    
    df_raw = None
    if file_path.endswith(('.xlsx', '.xls')):
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        target_sheet = sheet_names[0]
        for sheet in sheet_names:
            if 'data' in str(sheet).lower() or 'channel' in str(sheet).lower():
                target_sheet = sheet
                break
        
        preview = pd.read_excel(file_path, sheet_name=target_sheet, nrows=30, header=None)
        header_row = 0
        for idx, row in preview.iterrows():
            row_str = " ".join([str(val).lower() for val in row.values if pd.notna(val)])
            if any(term in row_str for term in ['volt', 'ecell', 'mv', 'time', 'duration']):
                header_row = idx
                break
        
        df_raw = pd.read_excel(file_path, sheet_name=target_sheet, skiprows=header_row)
    else:
        df_raw = pd.read_csv(file_path)

    # Clean column names to pure strings
    df_raw.columns = [str(col).strip() for col in df_raw.columns]

    # Detecting column names with expanded CALCE mappings
    col_map = {}
    for col in df_raw.columns:
        c_lower = col.lower()
        if 'time' in c_lower or 'duration' in c_lower:
            col_map[col] = 'Time [s]'
        elif 'volt' in c_lower or 'ecell' in c_lower or 'mv' in c_lower or c_lower == 'v':
            col_map[col] = 'Voltage [V]'
        elif 'curr' in c_lower or 'ma' in c_lower or 'i/a' in c_lower or c_lower == 'a':
            col_map[col] = 'Current [A]'
        elif 'cap' in c_lower or 'discharge' in c_lower or 'q_dis' in c_lower or 'mah' in c_lower:
            col_map[col] = 'Capacity [A.h]'

    df = df_raw.rename(columns=col_map).copy()

    # Convert Voltage from mV to V if detected as mV
    if 'Voltage [V]' in df.columns and df['Voltage [V]'].abs().max() > 100:
        df['Voltage [V]'] = df['Voltage [V]'] / 1000.0

    # Convert Current from mA to A if needed
    if 'Current [A]' in df.columns and df['Current [A]'].abs().max() > 100:
        df['Current [A]'] = df['Current [A]'] / 1000.0

    # Convert Capacity from mAh to Ah if needed
    if 'Capacity [A.h]' in df.columns and df['Capacity [A.h]'].abs().max() > 100:
        df['Capacity [A.h]'] = df['Capacity [A.h]'] / 1000.0

    if 'Time [s]' not in df.columns or 'Voltage [V]' not in df.columns:
        raise ValueError(f"Could not automatically detect Time/Voltage columns. Found columns: {list(df_raw.columns)[:5]}")

    # Selecting discharge phase
    if 'Current [A]' in df.columns:
        discharge_df = df[df['Current [A]'] < -1e-4].copy()
    else:
        discharge_df = df[df['Voltage [V]'].diff() <= 0].copy()

    if discharge_df.empty:
        discharge_df = df.copy()

    discharge_df = discharge_df.sort_values('Time [s]').reset_index(drop=True)

    # Capacity calculation
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

    required_cols = [
        'Time [s]', 'Voltage [V]', 'Capacity [A.h]',
        'Chemistry', 'Target_Capacity_Ah', 'Size_Multiplier',
        'SOH', 'Initial_SOC', 'Variation_ID', 'Ambient_Temperature_C'
    ]
    return discharge_df[required_cols]


### FUNCTION THAT MERGES ALL CALCE DATASETS INTO ONE CSV
def merge_calce_dataset(raw_input_dir, output_csv):
    """
    Finds CALCE files inside data/, processes each cell sample, and combines them.
    """
    all_traces = []

    # Updated patterns matching your exact CALCE dataset filenames
    dataset_configs = [
        {"pattern": "*A1-*", "chem": "LFP", "nominal_ah": 1.1},
        {"pattern": "*A123*", "chem": "LFP", "nominal_ah": 1.1},
        {"pattern": "*SP20*", "chem": "NMC", "nominal_ah": 2.0},
        {"pattern": "*INR*", "chem": "NMC", "nominal_ah": 2.0},
        {"pattern": "*20R*", "chem": "NMC", "nominal_ah": 2.0},
    ]

    variation_counter = 1
    processed_files = set()

    for config in dataset_configs:
        matching_files = glob.glob(os.path.join(raw_input_dir, "**", config["pattern"]), recursive=True)
        valid_files = [f for f in matching_files if f.endswith(('.xlsx', '.xls', '.csv'))]

        for fpath in valid_files:
            if fpath in processed_files:
                continue  # Avoid duplicate reads if file matches multiple patterns

            try:
                trace_df = parse_calce_discharge_file(
                    file_path=fpath,
                    chemistry=config["chem"],
                    target_capacity_ah=config["nominal_ah"],
                    variation_id=variation_counter
                )
                all_traces.append(trace_df)
                processed_files.add(fpath)
                variation_counter += 1
            except Exception as e:
                print(f"Skipping {fpath} due to error: {e}")

    if not all_traces:
        print(f"No files matching patterns found in '{raw_input_dir}'. Please verify your folder path.")
        return

    combined_df = pd.concat(all_traces, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"\n--> Successfully created merged raw dataset at: {output_csv}")
    print(f"Total rows: {len(combined_df)} | Total battery runs: {combined_df['Variation_ID'].nunique()}")

### MAIN EXECUTION
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    DOWNLOADS_DIR = os.path.join(REPO_ROOT, "data", "calce_experiment_data")
    OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "default", "raw", "advanced_synthetic_battery_data.csv")

    merge_calce_dataset(DOWNLOADS_DIR, OUTPUT_FILE)