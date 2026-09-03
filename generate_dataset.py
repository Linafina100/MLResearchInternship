import pandas as pd
import numpy as np

# Load the perfect synthetic data
base_df = pd.read_csv("synthetic_battery_data.csv")
lfp_base = base_df[base_df['Chemistry'] == 'LFP'].copy()
nmc_base = base_df[base_df['Chemistry'] == 'NMC'].copy()

augmented_data = []
battery_id = 1

# Function to create a mutated battery profile
def mutate_battery(df, chem, b_id):
    mutated = df.copy()
    
    # Simulate aging: stretch or shrink the capacity between 80% and 100% of original
    soh_factor = np.random.uniform(0.8, 1.0)
    mutated['Capacity [A.h]'] = mutated['Capacity [A.h]'] * soh_factor
    
    # Simulate Stena's sensor noise: add small random variations to the voltage
    noise = np.random.normal(0, 0.005, len(mutated))
    mutated['Voltage [V]'] = mutated['Voltage [V]'] + noise
    
    mutated['Battery_ID'] = b_id
    mutated['Chemistry'] = chem
    return mutated

# Generate 50 LFP and 50 NMC batteries
print("Generating 100 synthetic batteries...")
for _ in range(50):
    augmented_data.append(mutate_battery(lfp_base, "LFP", battery_id))
    battery_id += 1
    
for _ in range(50):
    augmented_data.append(mutate_battery(nmc_base, "NMC", battery_id))
    battery_id += 1

# Save the new large dataset
large_dataset = pd.concat(augmented_data)
large_dataset.to_csv("stena_training_data.csv", index=False)
print("Saved 100 batteries to 'stena_training_data.csv'!")