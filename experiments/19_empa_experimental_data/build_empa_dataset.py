import os
import glob
import json
import pandas as pd
import numpy as np

def parse_empa_cell(bdf_parquet_path):
    """
    Parses a single EMPA cell's time-series Parquet file and its companion JSON-LD metadata file.
    """
    base_path = bdf_parquet_path.split('.bdf.parquet')[0]
    meta_json_path = base_path + '.metadata.json'
    
    chemistry = "Unknown"
    cell_id = os.path.basename(base_path)
    
    if os.path.exists(meta_json_path):
        try:
            with open(meta_json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            graph = metadata.get('@graph', [])
            for item in graph:
                if item.get('@type') == 'BatteryTest':
                    pos_elec = item.get('hasTestObject', {}).get('hasPositiveElectrode', {})
                    active_mat = pos_elec.get('hasCoating', {}).get('hasActiveMaterial', {})
                    mat_comment = active_mat.get('rdfs:comment', '')
                    
                    if 'NickelCobaltManganese' in mat_comment or 'NMC' in mat_comment:
                        chemistry = 'NMC'
                    elif 'IronPhosphate' in mat_comment or 'LFP' in mat_comment:
                        chemistry = 'LFP'
        except Exception as e:
            print(f"Warning: Could not parse metadata for {cell_id}: {e}")

    # Read time-series Parquet file
    df = pd.read_parquet(bdf_parquet_path)

    # Map column names
    df['Time [s]'] = df['test_time_millisecond'] / 1000.0
    df['Voltage [V]'] = df['voltage_volt']
    df['Current [A]'] = df['current_ampere']
    
    # Calculate Capacity [A.h]
    dt = df['Time [s]'].diff().fillna(0) / 3600.0
    df['Capacity [A.h]'] = (df['Current [A]'].abs() * dt).cumsum()

    # Inject identifiers
    df['Chemistry'] = chemistry
    df['Variation_ID'] = cell_id
    df['Battery_ID'] = cell_id
    df['SOH'] = 1.0
    df['Initial_SOC'] = 1.0

    standard_cols = [
        'Time [s]', 'Voltage [V]', 'Capacity [A.h]', 'Current [A]', 
        'Chemistry', 'Variation_ID', 'Battery_ID', 'SOH', 'Initial_SOC', 'cycle_dimensionless'
    ]
    existing_cols = [c for c in standard_cols if c in df.columns]
    return df[existing_cols]

def build_empa_dataset(data_dir, output_parquet_path):
    search_pattern = os.path.join(data_dir, "**", "*.bdf.parquet")
    bdf_files = glob.glob(search_pattern, recursive=True)
    
    if not bdf_files:
        print(f"No BDF Parquet files found in directory: {data_dir}")
        return

    print(f"Found {len(bdf_files)} EMPA Parquet files. Processing...")

    all_traces = []
    total_files = len(bdf_files)
    
    for i, file_path in enumerate(bdf_files, 1):
        try:
            print(f"[{i}/{total_files}] Processing {os.path.basename(file_path)}...")
            cell_df = parse_empa_cell(file_path)
            all_traces.append(cell_df)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    if not all_traces:
        print("No data successfully parsed.")
        return

    print("Concatenating all datasets into master dataframe...")
    master_df = pd.concat(all_traces, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    
    # SAVE AS PARQUET INSTEAD OF CSV FOR BLISTERING SPEED
    print(f"Saving master dataset to Parquet: '{output_parquet_path}'...")
    master_df.to_parquet(output_parquet_path, index=False)
    print(f"Successfully finished! Saved {len(master_df)} total rows.")
    
if __name__ == "__main__":
    DATA_DIRECTORY = "data/empa_dataset/empa_dataset_raw"
    # Changed output extension to .parquet for speed and efficiency
    OUTPUT_FILE = "data/empa_dataset/empa_merged_dataset.parquet"
    
    build_empa_dataset(DATA_DIRECTORY, OUTPUT_FILE)