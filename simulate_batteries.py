import pybamm
import pandas as pd
import matplotlib.pyplot as plt

# Select the mathematical model (SPM = Single Particle Model)
model = pybamm.lithium_ion.SPM()

# Load chemical parameters for LFP and NMC from PyBaMM's built-in library
# "Prada2013" is a commonly used parameter set for LFP, while "Chen2020" is commonly used for NMC
param_lfp = pybamm.ParameterValues("Prada2013")
param_nmc = pybamm.ParameterValues("Chen2020")

# Define the "Stena experiment"
# We initially discharge to 2.5 V to ensure that the simulation runs without issues
experiment = pybamm.Experiment([
    "Discharge at 1C until 2.5 V"
])

# Set up the simulations
sim_lfp = pybamm.Simulation(model, parameter_values=param_lfp, experiment=experiment)
sim_nmc = pybamm.Simulation(model, parameter_values=param_nmc, experiment=experiment)

# 5. Run the simulations
print("Simulating LFP...")
solution_lfp = sim_lfp.solve()

print("Simulating NMC...")
solution_nmc = sim_nmc.solve()

# Extract the data for machine learning into Pandas DataFrames
lfp_data = pd.DataFrame({
    "Time [s]": solution_lfp["Time [s]"].entries,
    "Voltage [V]": solution_lfp["Terminal voltage [V]"].entries,
    "Capacity [A.h]": solution_lfp["Discharge capacity [A.h]"].entries,
    "Chemistry": "LFP"
})

nmc_data = pd.DataFrame({
    "Time [s]": solution_nmc["Time [s]"].entries,
    "Voltage [V]": solution_nmc["Terminal voltage [V]"].entries,
    "Capacity [A.h]": solution_nmc["Discharge capacity [A.h]"].entries,
    "Chemistry": "NMC"
})

# Combine the datasets and save them as a CSV file
training_data = pd.concat([lfp_data, nmc_data])
training_data.to_csv("synthetic_battery_data.csv", index=False)
print("Data sparad till 'synthetic_battery_data.csv'!")

# Plot the results to visually compare the voltage profiles of the two chemistries
pybamm.dynamic_plot([solution_lfp, solution_nmc], labels=["LFP", "NMC"])