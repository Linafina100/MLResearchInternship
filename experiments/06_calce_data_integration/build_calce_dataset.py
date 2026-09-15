"""This scripts automates teh process of finidng raw CALCE excel/csv files
in the data/calce_experiment_data/ folder, extracting the continous discharge
phases, normalizing column names and units, calculating capacity and SOH and
writing a unified dataset to data/default/raw/advanced_synthetic_battery_data.csv"""


### IMPORTS
import os
import glob
import pandas as pd
import numpy as np

### FUNCTION THAT PROCESSES ONE CALCE FILE
def parse_calce_discharge_file(file_path, chemistry, target_capacity_ah, variation_id):
    """
    Parses a single CALCE Excel/CSV file, identifies the C/20 continuous discharge step,
    calculates capacity and SOH, and formats columns for feature_engineering_continuous.py.
    """
    print(f"Processing ({chemistry}): {os.path.basename(file_path)}...")
    # Read the file
    if file_path.endswith(('.xlsx', '.xls')):
        df_raw = pd.read_excel(file_path)
    else:
        df_raw = pd.read_csv(file_path)
    #Detecting column names
    col_map = {}
    for col in df_raw.columns:
        c_lower = str(col).strip().lower()
        if 'time' in c_lower:
            col_map[col] = 'Time [s]'
        elif 'volt' in c_lower or c_lower == 'ecell/v':
            col_map[col] = 'Voltage [V]'
        elif 'curr' in c_lower or c_lower == '<i_mA>':
            col_map[col] = 'Current [A]'
        elif 'cap' in c_lower or 'discharge' in c_lower:
            col_map[col] = 'Capacity [A.h]'
    # Renaming columns and making a copy of the DataFrame
    df = df_raw.rename(columns=col_map).copy()
    # Converting units if necessary (e.g., mA to A, mAh to Ah)
    if 'Current [A]' in df.columns and df['Current [A]'].abs().max() > 100:
        df['Current [A특별시]' if 'Current [A특별시]' in df.columns else 'Current [A]'] = df['Current [A]'] / 1000.0
    if 'Capacity [A.h]' in df.columns and df['Capacity [A.h]'].abs().max() > 100:
        df['Capacity [A.h]'] = df['Capacity [A.h]'] / 1000.0
    # Checking for required columns
    if 'Time [s]' not in df.columns or 'Voltage [V]' not in df.columns:
        raise ValueError(f"Could not automatically detect Time/Voltage columns in {file_path}")
    # Selecting discharge phase based on current or voltage trend
    if 'Current [A]' in df.columns:
        discharge_df = df[df['Current [A]'] < -1e-4].copy()
    else:
        discharge_df = df[df['Voltage [V]'].diff() <= 0].copy()

    if discharge_df.empty:
        discharge_df = df.copy()

    discharge_df = discharge_df.sort_values('Time [s]').reset_index(drop=True)
    # Calculating capacity by integrating current over time if not already present
    if 'Capacity [A.h]' not in discharge_df.columns or discharge_df['Capacity [A.h]'].isna().all():
        dt = discharge_df['Time [s]'].diff().fillna(0)
        current_a = discharge_df['Current [A]'].abs() if 'Current [A]' in discharge_df.columns else 0.05 * target_capacity_ah
        discharge_df['Capacity [A.h]'] = (current_a * dt).cumsum() / 3600.0
    else:
        discharge_df['Capacity [A.h]'] = discharge_df['Capacity [A.h]'] - discharge_df['Capacity [A.h]'].iloc[0]
    # Calculate SOH based on the last capacity value and the target capacity
    actual_q_discharged = discharge_df['Capacity [A.h]'].iloc[-1]
    soh_val = round(float(actual_q_discharged / target_capacity_ah), 4)
    soh_val = min(1.0, max(0.5, soh_val))
    # Adding standardized metadata columns, for example initial SOC is assumed to be 100% at the start of discharge
    discharge_df['Chemistry'] = chemistry
    discharge_df['Target_Capacity_Ah'] = target_capacity_ah
    discharge_df['Size_Multiplier'] = 1.0
    discharge_df['SOH'] = soh_val
    # Assuming initial SOC is 100% at the start of discharge
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
    # Dataset configurations
    dataset_configs = [
        {"pattern": "*A123*", "chem": "LFP", "nominal_ah": 1.1},
        {"pattern": "*INR*", "chem": "NMC", "nominal_ah": 2.0},
        {"pattern": "*20R*", "chem": "NMC", "nominal_ah": 2.0},
    ]

    variation_counter = 1

    for config in dataset_configs:
        # The following line recursively searches for files matching the pattern in the specified directory
        matching_files = glob.glob(os.path.join(raw_input_dir, "**", config["pattern"]), recursive=True)
        valid_files = [f for f in matching_files if f.endswith(('.xlsx', '.xls', '.csv'))]

        for fpath in valid_files:
            try:
                # Each match assigns the target chemistry, nominal capacity and variation id per battery trace
                trace_df = parse_calce_discharge_file(
                    file_path=fpath,
                    chemistry=config["chem"],
                    target_capacity_ah=config["nominal_ah"],
                    variation_id=variation_counter
                )
                all_traces.append(trace_df)
                variation_counter += 1
            except Exception as e:
                print(f"Skipping {fpath} due to error: {e}")

    if not all_traces:
        print(f"No files matching patterns found in '{raw_input_dir}'. Please verify your folder path.")
        return

    combined_df = pd.concat(all_traces, ignore_index=True)
    # Saving the combined DataFrame to the specified output CSV path
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"\n--> Successfully created merged raw dataset at: {output_csv}")
    print(f"Total rows: {len(combined_df)} | Total battery runs: {combined_df['Variation_ID'].nunique()}")

### MAIN EXECUTION
"""The output merged dataset is saved to data/default/raw/advanced_synthetic_battery_data.csv
Dont know if this is the best place but I keep it for now"""
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    # Updated to point directly inside your data/ folder structure
    DOWNLOADS_DIR = os.path.join(REPO_ROOT, "data", "calce_experiment_data")
    OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "default", "raw", "advanced_synthetic_battery_data.csv")

    merge_calce_dataset(DOWNLOADS_DIR, OUTPUT_FILE)