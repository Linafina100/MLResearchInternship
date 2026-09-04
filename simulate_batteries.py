import pybamm
import pandas as pd
import matplotlib.pyplot as plt

# Select the mathematical model (SPM = Single Particle Model)
model = pybamm.lithium_ion.SPM()

# Load default chemical parameters for LFP and NMC
param_lfp_base = pybamm.ParameterValues("Prada2013")
param_nmc_base = pybamm.ParameterValues("Chen2020")

# Define the experiment (discharge to 2.5 V)
experiment = pybamm.Experiment(["Discharge at 1C until 2.5 V"])

# Multipliers to simulate different physical battery capacities
# E.g., 0.6 (~1.2 Ah), 1.0 (~2.0 Ah), and 1.75 (~3.5 Ah)
capacity_multipliers = [0.6, 1.0, 1.75]

# Empty list to store all generated data
all_data = []

print("Starting simulations for various battery sizes...")

for mult in capacity_multipliers:
    print(f"\n--- Simulating battery size with multiplier: {mult}x ---")
    
    # Create a clean copy of the base parameters for this specific iteration
    param_lfp = param_lfp_base.copy()
    param_nmc = param_nmc_base.copy()
    
    # Modify the electrode thicknesses to scale the total capacity
    param_lfp["Negative electrode thickness [m]"] *= mult
    param_lfp["Positive electrode thickness [m]"] *= mult
    
    param_nmc["Negative electrode thickness [m]"] *= mult
    param_nmc["Positive electrode thickness [m]"] *= mult
    
    # Build the simulations for this specific size
    sim_lfp = pybamm.Simulation(model, parameter_values=param_lfp, experiment=experiment)
    sim_nmc = pybamm.Simulation(model, parameter_values=param_nmc, experiment=experiment)
    
    # Run the solver
    print(f"Solving LFP ({mult}x)...")
    sol_lfp = sim_lfp.solve()
    
    print(f"Solving NMC ({mult}x)...")
    sol_nmc = sim_nmc.solve()
    
    # Extract LFP data
    lfp_data = pd.DataFrame({
        "Time [s]": sol_lfp["Time [s]"].entries,
        "Voltage [V]": sol_lfp["Terminal voltage [V]"].entries,
        "Capacity [A.h]": sol_lfp["Discharge capacity [A.h]"].entries,
        "Chemistry": "LFP",
        "Size_Multiplier": mult  # Track the physical size
    })
    all_data.append(lfp_data)
    
    # Extract NMC data
    nmc_data = pd.DataFrame({
        "Time [s]": sol_nmc["Time [s]"].entries,
        "Voltage [V]": sol_nmc["Terminal voltage [V]"].entries,
        "Capacity [A.h]": sol_nmc["Discharge capacity [A.h]"].entries,
        "Chemistry": "NMC",
        "Size_Multiplier": mult  # Track the physical size
    })
    all_data.append(nmc_data)

# Combine all results into a single DataFrame
training_data = pd.concat(all_data)

# Save to a new CSV file
output_file = "synthetic_battery_data_varied_sizes.csv"
training_data.to_csv(output_file, index=False)
print(f"\nDone! All data saved to '{output_file}'")