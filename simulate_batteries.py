"""
Simulates LFP/NMC battery discharge for the chemistry-classification
pipeline: a continuous constant-current discharge per battery, run to the
voltage cutoff, matching how Stena actually tests cells (no relaxation
rests) rather than a lab-style GITT pulse protocol.

Fixed 0.6C discharge rate for every battery, current scaled to each size
tier's own target capacity so 0.6C means the same thing at 1.2/2.0/3.5 Ah.
Chemistry pooling, capacity scaling, SOH, resistance/ambient-temperature
modeling, RANDOM_SEED, and the output-path convention are unchanged from
this project's earlier GITT-pulse-based version, archived at
experiments/07_pulse_protocol_archive/ (still used by
experiments/01_.../soc_sweep.py and
experiments/04_.../sweep_pulse_variable_discharge.py, which specifically
need the pulse protocol).

Usage: env vars SOC_RANGE_MIN/MAX, DATA_DIR, RUN_LABEL,
OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG (all optional,
overridable so sweep scripts can drive systematic experiments without
duplicating this file's setup).
"""
import os
import pybamm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

SOC_RANGE_MIN = float(os.environ.get("SOC_RANGE_MIN", 0.5))
SOC_RANGE_MAX = float(os.environ.get("SOC_RANGE_MAX", 1.0))

DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "default")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
DISCHARGE_PLOT_PNG = os.environ.get("DISCHARGE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "continuous_discharge_plot.png"))

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

model = pybamm.lithium_ion.SPM()

param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]

# Continuous discharge: fixed 0.6C for every battery, run to the voltage
# cutoff (no fixed time window). Current is scaled to each size tier's own
# target capacity so 0.6C means the same thing at 1.2/2.0/3.5 Ah.
DISCHARGE_C_RATE = 0.6
LOWER_VOLTAGE_CUTOFF = {
    "LFP": 1.8,
    "NMC": 2.3,
}


def make_continuous_discharge_experiment(current_a, v_min):
    """Build a single continuous constant-current discharge, no rests."""
    return pybamm.Experiment([f"Discharge at {current_a:.4f} A until {v_min} V"])


FAILURE_LOG = []


def solve_with_cutoff(sim, initial_soc, chem, context=""):
    """Solve an experiment; keep partial results if the voltage cut-off is hit.
    Any solve failure is logged.
    """
    try:
        sol = sim.solve(initial_soc=initial_soc)
    except pybamm.SolverError as err:
        sol = getattr(sim, "solution", None)
        recovered = sol is not None and len(sol.t) > 0
        print(f"  {chem}{context}: solver stopped early [SolverError] "
              f"({'partial data kept' if recovered else 'no data'}) ({err})")
        FAILURE_LOG.append({
            "Chemistry": chem, "Context": context,
            "ExceptionType": "SolverError", "Message": str(err), "Recovered": recovered,
        })
        if not recovered:
            return None
    except Exception as err:
        print(f"  {chem}{context}: solve failed [{type(err).__name__}] ({err})")
        FAILURE_LOG.append({
            "Chemistry": chem, "Context": context,
            "ExceptionType": type(err).__name__, "Message": str(err), "Recovered": False,
        })
        return None

    termination = getattr(sol, "termination", None)
    if termination and termination != "final time":
        print(f"  {chem}: terminated early ({termination})")

    return sol


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

variations_per_size = 83

all_data = []

print("Starting continuous-discharge simulations (Random SOC/SOH)...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soc = random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX)
        soh = random.uniform(0.50, 0.85)

        ambient_c = random.uniform(0, 35)
        TEMP_RESISTANCE_COEFF = 0.02
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)

        variation_id = size_idx * variations_per_size + i

        nmc_set_name = random.choice(NMC_PARAMETER_SETS)

        print(f"\n--- Target size: {target_ah} Ah | Variation {i+1}/{variations_per_size} | "
              f"SOC: {soc:.2f} | SOH: {soh:.2f} | Ambient: {ambient_c:.1f}C | NMC set: {nmc_set_name} ---")

        param_lfp = param_lfp_base.copy()
        param_nmc = param_nmc_bases[nmc_set_name].copy()

        chemistries = {
            "LFP": param_lfp,
            "NMC": param_nmc,
        }

        resistance_factors = {}
        for chem, param in chemistries.items():
            mult = capacity_multipliers[chem][size_idx]
            param["Negative electrode thickness [m]"] *= mult
            param["Positive electrode thickness [m]"] *= mult

            param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
            param["Maximum concentration in positive electrode [mol.m-3]"] *= soh

            resistance_noise = random.uniform(0.8, 1.2)
            resistance_factor = min(1.0, max(0.3, soh * resistance_noise * temp_conductivity_multiplier))
            resistance_factors[chem] = resistance_factor
            param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
            param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

        for chem, param in chemistries.items():
            v_min = LOWER_VOLTAGE_CUTOFF[chem]
            current_a = DISCHARGE_C_RATE * target_ah
            discharge_experiment = make_continuous_discharge_experiment(current_a, v_min)
            sim = pybamm.Simulation(model, parameter_values=param, experiment=discharge_experiment)

            print(f"Solving {chem} (I={current_a:.4f} A / 0.6C, Vmin={v_min} V)...")
            context = f" [target={target_ah}Ah, variation={i+1}/{variations_per_size}, SOH={soh:.3f}]"
            sol = solve_with_cutoff(sim, soc, chem, context=context)
            if sol is None:
                print(f"  Skipping {chem}: no solution data.")
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
                "V_min [V]": v_min,
                "Variation_ID": variation_id,
            })
            all_data.append(df)

# Combine and save
if all_data:
    training_data = pd.concat(all_data, ignore_index=True)
    training_data.to_csv(OUTPUT_DATA_CSV, index=False)
    print(f"\nDone! Continuous-discharge data with SOC and aging saved to '{OUTPUT_DATA_CSV}'")
else:
    training_data = pd.DataFrame()
    print("\nNo data generated: every solve attempt failed.")

# --- FAILURE SUMMARY ---
total_attempts = len(CAPACITY_TARGETS_AH) * variations_per_size * len(LOWER_VOLTAGE_CUTOFF)
print(f"\n{len(FAILURE_LOG)} of {total_attempts} solve attempts failed.")
if FAILURE_LOG:
    failures_df = pd.DataFrame(FAILURE_LOG)
    print(failures_df["ExceptionType"].value_counts().to_string())
    failures_df.to_csv(FAILURE_LOG_CSV, index=False)
    print(f"Full failure log saved to '{FAILURE_LOG_CSV}'")

# --- PLOTTING ---
# Overlay every run (not just the first LFP/NMC pair) so one failed early
# solve can't crash plotting, and the plot shows the full spread.
if not training_data.empty:
    print("Generating continuous discharge plot...")
    plt.figure(figsize=(12, 6))
    lfp_labeled, nmc_labeled = False, False
    for (chem, var_id), run_df in training_data.groupby(["Chemistry", "Variation_ID"]):
        color = '#1f77b4' if chem == "LFP" else '#ff7f0e'
        label = None
        if chem == "LFP" and not lfp_labeled:
            label, lfp_labeled = "LFP", True
        elif chem == "NMC" and not nmc_labeled:
            label, nmc_labeled = "NMC", True
        plt.plot(run_df['Time [s]'] / 3600, run_df['Voltage [V]'], color=color, alpha=0.15, linewidth=0.8, label=label)

    plt.title('Simulated Continuous Discharge Profiles (0.6C)')
    plt.xlabel('Time [Hours]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid(True)

    plt.savefig(DISCHARGE_PLOT_PNG, dpi=150)
    print(f"Plot saved to '{DISCHARGE_PLOT_PNG}'")
    plt.close()
else:
    print("Plot skipped: no data generated.")
