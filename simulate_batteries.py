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

# Pulse discharge (GITT) from the article, green/yellow identification interval:
# total discharged capacity is fixed at ΔQ = 0.6 Ah (half of the assumed
# minimum 18650 capacity of 1.2 Ah), independent of the actual cell size,
# each pulse lasts 0.5 h and is followed by a 1 h rest. The per-step current
# is I = 0.6 Ah / n_steps / 0.5 h, which reproduces the article's stated
# value of 80 mA for 15 steps, and gives 240/120/60/48 mA for 5/10/20/25
# steps respectively. All of these are <=0.2C for the smallest (1.2 Ah)
# reference cell, i.e. "generally small values" as required by the article
# so the terminal voltage stays inside the feasible green/yellow measurement
# zone ([max(Vmin)-0.2V, min(Vmax)+0.2V] = [2.3V, 3.8V] for LFP/NMC, Sec. 2)
# instead of overshooting into the orange/red (deep-discharge/overcharge)
# zones during a pulse.
# Nominal lower voltage limits (article Sec. 2): LFP 2.0 V, NMC 2.5 V.
TOTAL_PULSE_CAPACITY_AH = 0.6
PULSE_DURATION = "30 minutes"
PULSE_DURATION_HOURS = 0.5
REST_DURATION = "1 hour"
LOWER_VOLTAGE_CUTOFF = {
    "LFP": 2.0,
    "NMC": 2.5,
}
STEP_COUNTS = [5, 10, 15, 20, 25]


def pulse_current_ma(n_steps):
    """Per-pulse current [mA] so n_steps pulses discharge the fixed total
    of TOTAL_PULSE_CAPACITY_AH, per the article's GITT protocol."""
    return TOTAL_PULSE_CAPACITY_AH / n_steps / PULSE_DURATION_HOURS * 1000


def make_pulse_experiment(n_steps, v_min):
    """Build a GITT discharge experiment with n pulse/rest cycles."""
    pulse_current = f"{pulse_current_ma(n_steps):.4g} mA"
    return pybamm.Experiment(
        [
            (
                f"Discharge at {pulse_current} for {PULSE_DURATION} or until {v_min} V",
                f"Rest for {REST_DURATION}",
            )
        ]
        * n_steps
    )


def solve_with_cutoff(sim, initial_soc, chem, n_steps):
    """Solve an experiment; keep partial results if the voltage cut-off is hit."""
    try:
        sol = sim.solve(initial_soc=initial_soc)
    except pybamm.SolverError as err:
        sol = getattr(sim, "solution", None)
        print(f"  {chem} ({n_steps} steps): solver stopped early ({err})")
        if sol is None or len(sol.t) == 0:
            return None

    termination = getattr(sol, "termination", None)
    if termination and termination != "final time":
        print(f"  {chem} ({n_steps} steps): terminated early ({termination})")

    return sol


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

        chemistries = {
            "LFP": param_lfp,
            "NMC": param_nmc,
        }

        for n_steps in STEP_COUNTS:
            for chem, param in chemistries.items():
                v_min = LOWER_VOLTAGE_CUTOFF[chem]
                pulse_experiment = make_pulse_experiment(n_steps, v_min)
                sim = pybamm.Simulation(model, parameter_values=param, experiment=pulse_experiment)

                pulse_ma = pulse_current_ma(n_steps)
                print(f"Solving {chem} ({n_steps} steps, I={pulse_ma:.4g} mA, Vmin={v_min} V)...")
                sol = solve_with_cutoff(sim, soc, chem, n_steps)
                if sol is None:
                    print(f"  Skipping {chem} ({n_steps} steps): no solution data.")
                    continue

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
                    "Initial_SOC": round(soc, 3),
                    "N_Steps": n_steps,
                    "V_min [V]": v_min,
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
