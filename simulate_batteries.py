import pybamm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Select the mathematical model (SPM)
model = pybamm.lithium_ion.SPM()

# Load default chemical parameters
param_lfp_base = pybamm.ParameterValues("Prada2013")
param_nmc_base = pybamm.ParameterValues("Chen2020")

# Define the Pulse Discharge Experiment based on the article
# 15 steps of: 30 min discharge at 80 mA, followed by 1 hour of rest.
pulse_experiment = pybamm.Experiment(
    [
        (
            "Discharge at 80 mA for 30 minutes",
            "Rest for 1 hour"
        )
    ] * 15
)

# Multipliers for battery capacity sizing (~1.2Ah, ~2.0Ah, ~3.5Ah)
capacity_multipliers = [0.6, 1.0, 1.75]

# How many random variations to run per battery size
# Keep this low (e.g., 2) while testing, increase to generate massive datasets later
variations_per_size = 2 

all_data = []

print("Starting advanced simulations (Pulse Discharge, Random SOC/SOH)...")

for mult in capacity_multipliers:
    for i in range(variations_per_size):
        # Generate random SOC (50% to 100%) and SOH (80% to 100%)
        soc = random.uniform(0.5, 1.0)
        soh = random.uniform(0.8, 1.0)
        
        print(f"\n--- Size: {mult}x | Variation {i+1}/{variations_per_size} | SOC: {soc:.2f} | SOH: {soh:.2f} ---")
        
        # Create clean parameter copies
        param_lfp = param_lfp_base.copy()
        param_nmc = param_nmc_base.copy()
        
        # Apply Capacity Scaling (Electrode thickness)
        for param in [param_lfp, param_nmc]:
            param["Negative electrode thickness [m]"] *= mult
            param["Positive electrode thickness [m]"] *= mult
            
            # Apply Aging/SOH (Reduce maximum lithium concentration)
            param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
            param["Maximum concentration in positive electrode [mol.m-3]"] *= soh
            
        # Build Simulations
        sim_lfp = pybamm.Simulation(model, parameter_values=param_lfp, experiment=pulse_experiment)
        sim_nmc = pybamm.Simulation(model, parameter_values=param_nmc, experiment=pulse_experiment)
        
        # Solve with random starting SOC
        print("Solving LFP...")
        # Note: initial_soc tells the model how full it is before starting the experiment
        sol_lfp = sim_lfp.solve(initial_soc=soc) 
        
        print("Solving NMC...")
        sol_nmc = sim_nmc.solve(initial_soc=soc)
        
        # Extract data and apply Gaussian noise (standard deviation = 0.001)
        for chem, sol in [("LFP", sol_lfp), ("NMC", sol_nmc)]:
            raw_voltage = sol["Terminal voltage [V]"].entries
            noise = np.random.normal(0, 0.001, len(raw_voltage))
            voltage_with_noise = raw_voltage + noise
            
            df = pd.DataFrame({
                "Time [s]": sol["Time [s]"].entries,
                "Voltage [V]": voltage_with_noise,
                "Capacity [A.h]": sol["Discharge capacity [A.h]"].entries,
                "Chemistry": chem,
                "Size_Multiplier": mult,
                "SOH": round(soh, 3),
                "Initial_SOC": round(soc, 3)
            })
            all_data.append(df)

# Combine and save
training_data = pd.concat(all_data)
output_file = "advanced_synthetic_battery_data.csv"
training_data.to_csv(output_file, index=False)
print(f"\nDone! Data with realistic pulses, SOC, and aging saved to '{output_file}'")

# --- PLOTTING ---
# Plot one LFP and one NMC sample to visualize the pulse discharge
print("Generating pulse discharge plot...")

# all_data[0] is the first LFP simulation
# all_data[1] is the first NMC simulation
lfp_sample = all_data[0]
nmc_sample = all_data[1]

plt.figure(figsize=(12, 6))
plt.plot(lfp_sample['Time [s]'] / 3600, lfp_sample['Voltage [V]'], label='LFP (Sample)', color='#1f77b4')
plt.plot(nmc_sample['Time [s]'] / 3600, nmc_sample['Voltage [V]'], label='NMC (Sample)', color='#ff7f0e')

plt.title('Simulated Pulse Discharge Profiles (GITT)')
plt.xlabel('Time [Hours]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid(True)
plt.show()
