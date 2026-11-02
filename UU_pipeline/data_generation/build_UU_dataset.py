# Adapted for the NMC dataset
import os
import glob
import pandas as pd

def parse_time_to_seconds(time_str):
    """Converts NEWARE hh:mm:ss.ms time string to total seconds."""
    if pd.isna(time_str):
        return 0.0
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except Exception:
        pass
    return 0.0

def process_neware_excel(file_path, chemistry_label, cell_counter):
    # Read the record sheet from the Neware Excel file
    df = pd.read_excel(file_path, sheet_name="record")
    
    # Standardize column names & units
    processed_df = pd.DataFrame()
    processed_df['Chemistry'] = chemistry_label
    processed_df['Size_Multiplier'] = 1.0
    processed_df['SOH'] = 1.0
    processed_df['Initial_SOC'] = 1.0
    processed_df['Variation_ID'] = cell_counter
    
    # Map time, voltage, capacity
    processed_df['Time [spy]'] = df['Total Time'].apply(parse_time_to_seconds) if 'Total Time' in df.columns else 0.0
    # Note: feature_engineering.py looks for 'Time [s]'
    processed_df['Time [s]'] = df['Total Time'].apply(parse_time_to_seconds)
    processed_df['Voltage [V]'] = df['Voltage(V)']
    processed_df['Capacity [A.h]'] = df['Capacity(mAh)'] / 1000.0 # Convert mAh to Ah
    processed_df['Cycle_Index'] = df['Cycle Index']
    
    return processed_df

def build_combined_csv(nmc_dir, lfp_dir, output_csv):
    all_data = []
    cell_id = 1
    
    # Process all NMC files
    for filepath in glob.glob(os.path.join(nmc_dir, "*.xlsx")):
        print(f"Processing NMC cell: {filepath}")
        df_cell = process_neware_excel(filepath, "NMC", cell_id)
        all_data.append(df_cell)
        cell_id += 1
        
    # Process all LFP files
    for filepath in glob.glob(os.path.join(lfp_dir, "*.xlsx")):
        print(f"Processing LFP cell: {filepath}")
        df_cell = process_neware_excel(filepath, "LFP", cell_id)
        all_data.append(df_cell)
        cell_id += 1
        
    if not all_data:
        print("No Excel files found in specified directories!")
        return
        
    combined = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"Successfully created combined experimental file at: {output_csv} ({len(combined)} rows)")

if __name__ == "__main__":
    NMC_SOURCE_DIR = "data/raw_experimental/NMC"
    LFP_SOURCE_DIR = "data/raw_experimental/LFP"
    OUTPUT_FILE = "UU_pipeline/data/combined_uu_experimental_data.csv"
    
    build_combined_csv(NMC_SOURCE_DIR, LFP_SOURCE_DIR, OUTPUT_FILE)