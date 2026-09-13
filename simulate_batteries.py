import os
import pybamm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Fixed seed: makes a run (including which draws fail to solve) reproducible for debugging.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Starting-SOC range, overridable via env vars so sweep scripts can test
# different SOC availability without duplicating this file's setup.
# Default 50-100% matches the article.
SOC_RANGE_MIN = float(os.environ.get("SOC_RANGE_MIN", 0.5))
SOC_RANGE_MAX = float(os.environ.get("SOC_RANGE_MAX", 1.0))

# All outputs live under data/<run_label>/<kind>/ so different runs and file
# kinds never collide. RUN_LABEL defaults to "default"; sweep scripts
# override it per run (e.g. "soc_0.1-0.4").
DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "default")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

# Individually overridable if an exact path is ever needed.
OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
PULSE_PLOT_PNG = os.environ.get("PULSE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "pulse_discharge_plot.png"))

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, PULSE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

# Isothermal SPM: the thermal submodel needs entropic-heat data Prada2013
# doesn't define, and doesn't actually wire ambient temperature into
# kinetics anyway (verified: sweeping 0-35C gave identical output).
# Temperature's real effect is applied manually below via
# TEMP_RESISTANCE_COEFF instead.
model = pybamm.lithium_ion.SPM()

# NMC pooled across 3 parameter sets so the model can't memorize one OCP
# curve as a chemistry tell. ORegan2022 excluded: its conductivity is a
# function, not a scalar, incompatible with the *= scaling below.
param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]  # reference for capacity_multipliers below

# GITT protocol from the article: total discharge dQ=0.6Ah split evenly
# across n_steps pulses (0.5h each + 1h rest) keeps current <=0.2C and
# voltage inside the feasible measurement zone. Lower cutoffs (article
# Sec. 2): LFP 2.0V, NMC 2.5V.
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


# Collects every solve failure (which chemistry/step-count/size/SOH, and what
# exception was raised) so the sample-count deficit between the theoretical
# chemistries*sizes*variations target and what actually lands in the CSV can
# be diagnosed after the run instead of only being visible as a missing row.
FAILURE_LOG = []


def solve_with_cutoff(sim, initial_soc, chem, n_steps, context=""):
    """Solve an experiment; keep partial results if the voltage cut-off is hit.

    Any solve failure is logged (with its exception type) and skipped rather
    than left to propagate. Previously only pybamm.SolverError was caught, so
    any other exception type raised by sim.solve() (e.g. from an extreme
    scaled-parameter combination) would crash the entire generation run
    without a trace, silently losing every remaining (chemistry, n_steps)
    attempt for that variation.
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


# Target capacities from the article (1.2/2.0/3.5 Ah). LFP and NMC have very
# different base capacities, so each gets its own thickness multiplier to
# actually hit the same targets -- keeps capacity itself uninformative
# about chemistry.
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

# 83 variations x 3 sizes = 249 samples/chemistry, matching the article's
# target of 250.
variations_per_size = 83

all_data = []

print("Starting advanced simulations (Pulse Discharge, Random SOC/SOH)...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        # SOC: uniform over the configurable range (default 50-100%,
        # matching the article's own sampling). SOH: uniform 50-85%,
        # matching Stena's degraded end-of-life population rather than a
        # healthier range never seen at inference.
        soc = random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX)
        soh = random.uniform(0.50, 0.85)

        # Ambient temperature, independent of SOH: real cells at equal SOH
        # show different resistance depending on aging pathway and storage
        # conditions. Applied via TEMP_RESISTANCE_COEFF below.
        ambient_c = random.uniform(0, 35)
        TEMP_RESISTANCE_COEFF = 0.02  # ~2%/C, a conservative literature estimate
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)

        # Unique id per (size, variation) draw. SOH is stored rounded to 3
        # decimals, so distinct draws can round-collide; Variation_ID keeps
        # them distinct in every downstream groupby.
        variation_id = size_idx * variations_per_size + i

        # Pool NMC across independent literature parameter sets so the model
        # can't memorize one fixed OCP curve as a chemistry signature.
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

            # 2. Resistance growth: decoupled from SOH via an independent
            # noise draw (aging-pathway variance) plus the shared
            # ambient-temperature effect, clamped to [0.3, 1.0] to avoid
            # exceeding the unaged baseline or a solver-destabilizing
            # near-zero.
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
training_data.to_csv(OUTPUT_DATA_CSV, index=False)
print(f"\nDone! Data with realistic pulses, SOC, and aging saved to '{OUTPUT_DATA_CSV}'")

# --- FAILURE SUMMARY ---
# How much of the theoretical target was lost to solve failures, and why.
total_attempts = len(CAPACITY_TARGETS_AH) * variations_per_size * len(STEP_COUNTS) * len(LOWER_VOLTAGE_CUTOFF)
print(f"\n{len(FAILURE_LOG)} of {total_attempts} solve attempts failed.")
if FAILURE_LOG:
    failures_df = pd.DataFrame(FAILURE_LOG)
    print(failures_df["ExceptionType"].value_counts().to_string())
    failures_df.to_csv(FAILURE_LOG_CSV, index=False)
    print(f"Full failure log saved to '{FAILURE_LOG_CSV}'")

# --- PLOTTING ---
# Plot one LFP and one NMC sample to visualize the pulse discharge
print("Generating pulse discharge plot...")

# all_data[0]/[1] are the first LFP/NMC simulations
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

plt.savefig(PULSE_PLOT_PNG, dpi=150)
print(f"Plot saved to '{PULSE_PLOT_PNG}'")

plt.show()
