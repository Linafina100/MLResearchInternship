"""
Variable-discharge-rate variant of simulate_batteries.py, for the
pulse-protocol + randomized-discharge-rate + SOC-sweep experiment in this
folder.

Difference from the live simulate_batteries.py: there, per-pulse current
is entirely a function of step count (a fixed TOTAL_PULSE_CAPACITY_AH=0.6Ah
spread evenly across n_steps pulses), not a property of the battery
itself. Here, each battery instead draws its own random C-rate
(uniform 0.1C-0.5C, scaled to its own target capacity) once per variation,
applied identically across all step counts -- so unlike the live script, a
battery's 25-step trace now discharges proportionally more total capacity
than its 5-step trace, rather than the same fixed 0.6Ah either way. Pulse
*timing* (30min pulse + 1h rest per step) is unchanged, so
feature_engineering.py's step-boundary detection (which keys off timing,
not current magnitude) works unmodified -- see this folder's
sweep_pulse_variable_discharge.py, which imports feature_engineering.py
and ml_pipeline.py directly from the repo root.

Every other physics/degradation choice (chemistry pooling, capacity
scaling, SOH, resistance/ambient-temperature modeling, RANDOM_SEED,
output-path convention) is unchanged -- see simulate_batteries.py's own
comments for the rationale behind those.

Usage: same env vars as simulate_batteries.py (SOC_RANGE_MIN/MAX,
DATA_DIR, RUN_LABEL, OUTPUT_DATA_CSV, FAILURE_LOG_CSV).
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
PULSE_PLOT_PNG = os.environ.get("PULSE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "pulse_discharge_plot.png"))

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, PULSE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

model = pybamm.lithium_ion.SPM()

param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]

# GITT pulse timing unchanged from simulate_batteries.py (0.5h pulse, 1h
# rest per step) -- only the current magnitude varies now, per battery.
PULSE_DURATION = "30 minutes"
REST_DURATION = "1 hour"
LOWER_VOLTAGE_CUTOFF = {
    "LFP": 1.8,
    "NMC": 2.3,
}
STEP_COUNTS = [5, 10, 15, 20, 25]

# Per-battery discharge rate, replacing the step-count-derived current:
# each battery draws its own C-rate once, used for every step count.
PULSE_C_RATE_MIN = 0.1
PULSE_C_RATE_MAX = 0.5


def make_pulse_experiment(current_a, n_steps, v_min):
    """Build a GITT discharge experiment with n pulse/rest cycles at a
    fixed current (this battery's own randomly-drawn C-rate)."""
    pulse_current = f"{current_a * 1000:.4g} mA"
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

print("Starting variable-discharge-rate pulse simulations (Random SOC/SOH/C-rate)...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soc = random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX)
        soh = random.uniform(0.50, 0.85)

        # Per-battery discharge rate: drawn once per variation, applied to
        # every step count for this battery (both chemistries use their
        # own current, scaled to their own target capacity, at this same
        # C-rate).
        pulse_c_rate = random.uniform(PULSE_C_RATE_MIN, PULSE_C_RATE_MAX)

        ambient_c = random.uniform(0, 35)
        TEMP_RESISTANCE_COEFF = 0.02
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)

        variation_id = size_idx * variations_per_size + i

        nmc_set_name = random.choice(NMC_PARAMETER_SETS)

        print(f"\n--- Target size: {target_ah} Ah | Variation {i+1}/{variations_per_size} | "
              f"SOC: {soc:.2f} | SOH: {soh:.2f} | C-rate: {pulse_c_rate:.3f} | "
              f"Ambient: {ambient_c:.1f}C | NMC set: {nmc_set_name} ---")

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

        for n_steps in STEP_COUNTS:
            for chem, param in chemistries.items():
                v_min = LOWER_VOLTAGE_CUTOFF[chem]
                current_a = pulse_c_rate * target_ah
                pulse_experiment = make_pulse_experiment(current_a, n_steps, v_min)
                sim = pybamm.Simulation(model, parameter_values=param, experiment=pulse_experiment)

                print(f"Solving {chem} ({n_steps} steps, I={current_a * 1000:.4g} mA, Vmin={v_min} V)...")
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
                    "Pulse_C_Rate": round(pulse_c_rate, 4),
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
training_data.to_csv(OUTPUT_DATA_CSV, index=False)
print(f"\nDone! Data with variable-rate pulses, SOC, and aging saved to '{OUTPUT_DATA_CSV}'")

# --- FAILURE SUMMARY ---
total_attempts = len(CAPACITY_TARGETS_AH) * variations_per_size * len(STEP_COUNTS) * len(LOWER_VOLTAGE_CUTOFF)
print(f"\n{len(FAILURE_LOG)} of {total_attempts} solve attempts failed.")
if FAILURE_LOG:
    failures_df = pd.DataFrame(FAILURE_LOG)
    print(failures_df["ExceptionType"].value_counts().to_string())
    failures_df.to_csv(FAILURE_LOG_CSV, index=False)
    print(f"Full failure log saved to '{FAILURE_LOG_CSV}'")

# --- PLOTTING ---
print("Generating pulse discharge plot...")

# all_data[0]/[1] are the first LFP/NMC simulations
lfp_sample = all_data[0]
nmc_sample = all_data[1]

plt.figure(figsize=(12, 6))
plt.plot(lfp_sample['Time [s]'] / 3600, lfp_sample['Voltage [V]'], label='LFP (Sample)', color='#1f77b4')
plt.plot(nmc_sample['Time [s]'] / 3600, nmc_sample['Voltage [V]'], label='NMC (Sample)', color='#ff7f0e')

plt.title('Simulated Pulse Discharge Profiles (Variable C-rate GITT)')
plt.xlabel('Time [Hours]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid(True)

plt.savefig(PULSE_PLOT_PNG, dpi=150)
print(f"Plot saved to '{PULSE_PLOT_PNG}'")

plt.show()
