import os
import pybamm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Fixed seed for reproducable runs
random.seed(42)
np.random.seed(42)

# Starting-SOC range, overridable via env vars so sweep scripts can test
# different SOC availability without duplicating this file's setup.
SOC_RANGE_MIN = float(os.environ.get("SOC_RANGE_MIN", 0.5))
SOC_RANGE_MAX = float(os.environ.get("SOC_RANGE_MAX", 1.0))

# ---- CREATING STRUCTURE FOR FILES ---

# Create a dynamic output directory structure to prevent different
# test runs (e.g., from run_soc_sweep.py) from overwriting each other.
DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "default")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

# Define file paths for data, logs, and plots. 
# Sorted by type inside the run folder by default, overridable if needed
OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
PULSE_PLOT_PNG = os.environ.get("PULSE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "pulse_discharge_plot.png"))

# Create the required folders before saving files to prevent errors
for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, PULSE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

# --- LOADING PYBAMM MODELS ---

# Use the isothermal SPM model. PyBaMM's thermal submodel is disabled 
# because the required entropic-heat parameters are undefined for LFP (Prada2013).
# Instead of guessing thermal parameters, the effect of temperature on internal 
# resistance is applied manually below via TEMP_RESISTANCE_COEFF.
model = pybamm.lithium_ion.SPM()

# Load base chemical parameters.
# LFP uses a single standard set (Prada). NMC pools three independent datasets 
# (Chen 811, Mohtat 532, and OKane 811/111-aging) to prevent the ML model from 
# memorizing a single specific chemistry variation.
# (ORegan2022 excluded due to incompatibility with scalar conductivity scaling)
param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"] 

# --- PULSE EXPERIMENT ---

# GITT pulse discharge parameters from the reference article.
# Discharges a fixed total of 0.6 Ah using small currents
# divided into various step counts (0.5h pulse, 1h rest). 
# Small currents ensure the voltage stays in the safe measurement zone 
TOTAL_PULSE_CAPACITY_AH = 0.6
PULSE_DURATION = "30 minutes"
PULSE_DURATION_HOURS = 0.5
REST_DURATION = "1 hour"
LOWER_VOLTAGE_CUTOFF = {
    "LFP": 1.8,
    "NMC": 2.3,
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

FAILURE_LOG = []


def solve_with_cutoff(sim, initial_soc, chem, n_steps, context=""):
    """Solve an experiment; keep partial results if the voltage cut-off is hit.
    Any solve failure is logged.
    """
    try:
        sol = sim.solve(initial_soc=initial_soc)
    except pybamm.SolverError as err:
        sol = getattr(sim, "solution", None)
        recovered = sol is not None and len(sol.t) > 0
        print(f"  {chem} ({n_steps} steps){context}: solver stopped early [SolverError] "
              f"({'partial data kept' if recovered else 'no data'}) ({err})")
        FAILURE_LOG.append({
            "Chemistry": chem, "N_Steps": n_steps, "Context": context,
            "ExceptionType": "SolverError", "Message": str(err), "Recovered": recovered,
        })
        if not recovered:
            return None
    except Exception as err:
        print(f"  {chem} ({n_steps} steps){context}: solve failed [{type(err).__name__}] ({err})")
        FAILURE_LOG.append({
            "Chemistry": chem, "N_Steps": n_steps, "Context": context,
            "ExceptionType": type(err).__name__, "Message": str(err), "Recovered": False,
        })
        return None

    termination = getattr(sol, "termination", None)
    if termination and termination != "final time":
        print(f"  {chem} ({n_steps} steps): terminated early ({termination})")

    return sol


# Target cell capacities representing standard 18650 cell sizes (1.2, 2.0, 3.5 Ah).
# Per-chemistry multipliers are calculated below to scale LFP (base ~2.3 Ah) 
# and NMC (base ~5.0 Ah) to these exact targets. This ensures capacity 
# remains identical across chemistries, preventing ML data leakage.
CAPACITY_TARGETS_AH = [1.2, 2.0, 3.5]


def capacity_multipliers_for(base_params, targets_ah):
    """Per-chemistry thickness multipliers that scale a base parameter set's
    own nominal capacity onto each of the target capacities."""
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    return [target_ah / base_capacity_ah for target_ah in targets_ah]


capacity_multipliers = {
    "LFP": capacity_multipliers_for(param_lfp_base, CAPACITY_TARGETS_AH),
    "NMC": capacity_multipliers_for(param_nmc_base, CAPACITY_TARGETS_AH),
}

# How many random variations to run per battery size. The article (Sec. 3)
# generates 250 individual OCV samples per chemistry for each tested
# capacity/step configuration. 
variations_per_size = 83

all_data = []

print("Starting advanced simulations (Pulse Discharge, Random SOC/SOH)...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soc = random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX)
        soh = random.uniform(0.50, 0.85)

        # Randomize ambient temperature (0-35°C). 
        # Since PyBaMM's thermal model is disabled, temperature effects on internal resistance 
        # are modeled manually. Conductivity shifts by ~2% per degree from a 25°C baseline.
        # This decouples IR-drop from SOH, simulating diverse real-world aging pathways.
        ambient_c = random.uniform(0, 35)
        TEMP_RESISTANCE_COEFF = 0.02 
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)

        # Create a unique ID for each battery.
        variation_id = size_idx * variations_per_size + i

        # Randomly select one of the three NMC profiles for this specific battery.
        nmc_set_name = random.choice(NMC_PARAMETER_SETS)

        print(f"\n--- Target size: {target_ah} Ah | Variation {i+1}/{variations_per_size} | "
              f"SOC: {soc:.2f} | SOH: {soh:.2f} | Ambient: {ambient_c:.1f}C | NMC set: {nmc_set_name} ---")

        # Create clean parameter copies
        param_lfp = param_lfp_base.copy()
        param_nmc = param_nmc_bases[nmc_set_name].copy()

        chemistries = {
            "LFP": param_lfp,
            "NMC": param_nmc,
        }

        # Apply per-chemistry Capacity Scaling (Electrode thickness) and Aging/SOH
        resistance_factors = {}
        for chem, param in chemistries.items():
            mult = capacity_multipliers[chem][size_idx]
            param["Negative electrode thickness [m]"] *= mult
            param["Positive electrode thickness [m]"] *= mult

            # 1. Capacity loss (LAM): reduce max lithium concentration
            param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
            param["Maximum concentration in positive electrode [mol.m-3]"] *= soh

            # 2. Resistance growth.
            # Multiplies SOH with random noise (0.8-1.2) and the temperature effect.
            # This ensures cells with the same SOH have unique internal resistance (IR-drop).
            # Clamped between 0.3 and 1.0 to prevent impossible values or solver crashes.
            resistance_noise = random.uniform(0.8, 1.2)
            resistance_factor = min(1.0, max(0.3, soh * resistance_noise * temp_conductivity_multiplier))
            resistance_factors[chem] = resistance_factor
            param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
            param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

        for n_steps in STEP_COUNTS:
            for chem, param in chemistries.items():
                v_min = LOWER_VOLTAGE_CUTOFF[chem]
                pulse_experiment = make_pulse_experiment(n_steps, v_min)
                sim = pybamm.Simulation(model, parameter_values=param, experiment=pulse_experiment)

                pulse_ma = pulse_current_ma(n_steps)
                print(f"Solving {chem} ({n_steps} steps, I={pulse_ma:.4g} mA, Vmin={v_min} V)...")
                context = f" [target={target_ah}Ah, variation={i+1}/{variations_per_size}, SOH={soh:.3f}]"
                sol = solve_with_cutoff(sim, soc, chem, n_steps, context=context)
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
                    "Target_Capacity_Ah": target_ah,
                    "Size_Multiplier": capacity_multipliers[chem][size_idx],
                    "SOH": round(soh, 3),
                    "Initial_SOC": round(soc, 3),
                    "Ambient_Temperature_C": round(ambient_c, 2),
                    "Resistance_Factor": round(resistance_factors[chem], 4),
                    "Base_Parameter_Set": nmc_set_name if chem == "NMC" else "Prada2013",
                    "N_Steps": n_steps,
                    "V_min [V]": v_min,
                    "Variation_ID": variation_id,
                })
                all_data.append(df)

# Combine and save
training_data = pd.concat(all_data)
output_file = OUTPUT_DATA_CSV
training_data.to_csv(output_file, index=False)
print(f"\nDone! Data with realistic pulses, SOC, and aging saved to '{output_file}'")

# --- FAILURE SUMMARY ---
# Surfaces exactly how much of the theoretical chemistries*sizes*variations
# target was lost to solve failures, and why, instead of that deficit only
# showing up later as an unexplained gap in the feature-engineered dataset.
total_attempts = len(CAPACITY_TARGETS_AH) * variations_per_size * len(STEP_COUNTS) * len(LOWER_VOLTAGE_CUTOFF)
print(f"\n{len(FAILURE_LOG)} of {total_attempts} solve attempts failed.")
if FAILURE_LOG:
    failures_df = pd.DataFrame(FAILURE_LOG)
    print(failures_df["ExceptionType"].value_counts().to_string())
    failure_log_file = FAILURE_LOG_CSV
    failures_df.to_csv(failure_log_file, index=False)
    print(f"Full failure log saved to '{failure_log_file}'")

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

plot_file = PULSE_PLOT_PNG
plt.savefig(plot_file, dpi=150)
print(f"Plot saved to '{plot_file}'")

plt.show()
