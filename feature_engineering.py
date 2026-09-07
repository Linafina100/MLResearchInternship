import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_features(input_csv, output_csv):
    print("Loading raw simulation data...")
    df = pd.read_csv(input_csv)
    
    # Create a unique ID for each simulated battery run
    # Since we generated multiple sizes and variations, we group them by their unique traits
    df['Battery_ID'] = df.groupby(['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC']).ngroup()
    
    all_features = []
    
    for battery_id, group in df.groupby('Battery_ID'):
        # Ensure time is perfectly sorted
        group = group.sort_values('Time [s]').reset_index(drop=True)
        
        # Identify the resting phases (OCV)
        # Calculate the change in capacity between each row
        group['Delta_Cap'] = group['Capacity [A.h]'].diff().fillna(0)
        
        # The battery is resting when capacity is not changing
        group['Is_Resting'] = group['Delta_Cap'] < 1e-6
        
        # We want the absolute last point of the rest phase, right before the next pulse starts.
        # Shift(-1) looks at the boolean value of the NEXT row.
        group['Next_Is_Discharging'] = ~(group['Is_Resting'].shift(-1).fillna(False))
        
        # Extract the perfect OCV points (e.g., 15 points if 15 pulses were completed)
        ocv_points = group[group['Is_Resting'] & group['Next_Is_Discharging']].copy()
        
        # Calculate discrete dV/dQ between these resting points
        ocv_points['dV'] = ocv_points['Voltage [V]'].diff()
        ocv_points['dQ'] = ocv_points['Capacity [A.h]'].diff()
        
        # dV/dQ calculation (fill the first step with 0 since diff() yields NaN)
        ocv_points['dV_dQ'] = (ocv_points['dV'] / ocv_points['dQ']).fillna(0)
        
        # Format the features for Scikit-Learn
        # Extract the dV/dQ values as a flat array 
        dvdq_values = ocv_points['dV_dQ'].values
        
        # Save metadata and labels
        battery_features = {
            'Battery_ID': battery_id,
            'Chemistry': group['Chemistry'].iloc[0],
            'Size_Multiplier': group['Size_Multiplier'].iloc[0],
            'SOH': group['SOH'].iloc[0],
            'Initial_SOC': group['Initial_SOC'].iloc[0]
        }
        
        # Dynamically create columns for each dV/dQ step (dV_dQ_step_1, dV_dQ_step_2, etc.)
        for step_idx, val in enumerate(dvdq_values):
            battery_features[f'dV_dQ_step_{step_idx+1}'] = val
            
        all_features.append(battery_features)
        
    # Save the cleaned feature matrix
    features_df = pd.DataFrame(all_features)
    
    # Fill any missing steps (if a battery hit the voltage cut-off early and didn't complete 15 steps)
    features_df = features_df.fillna(0)
    
    features_df.to_csv(output_csv, index=False)
    print(f"\nFeature engineering complete! {len(features_df)} battery profiles saved to '{output_csv}'")
    
    return features_df

if __name__ == "__main__":
    # Run the extraction
    input_file = "advanced_synthetic_battery_data.csv"
    output_file = "ml_features_data.csv"
    features = create_features(input_file, output_file)
    
    # Print the first few rows to verify the output shape
    print("\nPreview of ML features:")
    print(features.head())
    # --- PLOTTING ---

    print("Generating dV/dQ feature plot...")
    
    # Grab the first LFP and the first NMC row from the features dataframe
    lfp_sample = features[features['Chemistry'] == 'LFP'].iloc[0]
    nmc_sample = features[features['Chemistry'] == 'NMC'].iloc[0]

    # Extract just the dV/dQ step columns
    step_cols = [col for col in features.columns if col.startswith('dV_dQ_step_')]
    
    # Create the x-axis (Step 1 to 15)
    steps = range(1, len(step_cols) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, lfp_sample[step_cols], marker='o', label='LFP dV/dQ', linewidth=2)
    plt.plot(steps, nmc_sample[step_cols], marker='s', label='NMC dV/dQ', linewidth=2)

    plt.title('Extracted dV/dQ Features per Resting Step')
    plt.xlabel('OCV Measurement Step')
    plt.ylabel('dV/dQ [V/Ah]')
    plt.xticks(steps)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()