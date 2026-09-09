import pybamm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Fixed seed so a generation run (including which random SOH/size draws, if
# any, fail to solve) is reproducible run-to-run instead of silently varying
# every time, which previously made sample-count deficits impossible to
# investigate after the fact.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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


# Target cell capacities from the article (Sec. 3): "the electrode geometries
# are adjusted to different capacities for training purposes, i.e., 1.2 Ah,
# 2 Ah, and 3.5 Ah, covering the common capacity range of 18650 cells."
# LFP (Prada2013, base ~2.3 Ah) and NMC (Chen2020, base ~5.0 Ah) have very
# different base capacities, so a single shared thickness multiplier does not
# land both chemistries on the same target capacity. Instead, a per-chemistry
# multiplier is derived from each base parameter set's own declared
# "Nominal cell capacity [A.h]", so that both chemistries actually reach
# ~1.2/2.0/3.5 Ah at each size step, keeping capacity itself uninformative
# about chemistry (as intended by the article).
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
# capacity/step configuration. With 2 chemistries (LFP, NMC) and 3 capacity
# sizes here, variations_per_size = 83 gives 3*83 = 249 samples per chemistry,
# matching that target (~10 min runtime measured for the full sweep).
variations_per_size = 83

all_data = []

print("Starting advanced simulations (Pulse Discharge, Random SOC/SOH)...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        # SOC is fixed at 100% for now; SOH is randomized over a narrower
        # 75-85% aged range (per teammate update in ella_branch)
        soc = 1.0
        soh = random.uniform(0.75, 0.85)

        # Globally unique id for this (size, variation) draw. SOH is stored
        # rounded to 3 decimals, and with 83 random draws per size from only
        # ~101 possible rounded values, collisions are near-certain (birthday
        # paradox) -- two *different* variations can round to the identical
        # SOH, and since Initial_SOC and Size_Multiplier are otherwise
        # constant per size, that made them indistinguishable to every
        # downstream (Chemistry, Size_Multiplier, SOH, Initial_SOC) groupby,
        # silently merging independent simulation runs into one battery
        # group. Variation_ID is unaffected by any rounding and guarantees
        # each (size, i) draw stays its own identity everywhere downstream.
        variation_id = size_idx * variations_per_size + i

        print(f"\n--- Target size: {target_ah} Ah | Variation {i+1}/{variations_per_size} | SOC: {soc:.2f} | SOH: {soh:.2f} ---")

        # Create clean parameter copies
        param_lfp = param_lfp_base.copy()
        param_nmc = param_nmc_base.copy()

        chemistries = {
            "LFP": param_lfp,
            "NMC": param_nmc,
        }

        # Apply per-chemistry Capacity Scaling (Electrode thickness) and Aging/SOH
        for chem, param in chemistries.items():
            mult = capacity_multipliers[chem][size_idx]
            param["Negative electrode thickness [m]"] *= mult
            param["Positive electrode thickness [m]"] *= mult

            # 1. Capacity loss (LAM): reduce max lithium concentration
            param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
            param["Maximum concentration in positive electrode [mol.m-3]"] *= soh

            # 2. Resistance growth: decrease electrode conductivity as SOH decreases
            param["Negative electrode conductivity [S.m-1]"] *= soh
            param["Positive electrode conductivity [S.m-1]"] *= soh

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
                    "N_Steps": n_steps,
                    "V_min [V]": v_min,
                    "Variation_ID": variation_id,
                })
                all_data.append(df)

# Combine and save
training_data = pd.concat(all_data)
output_file = "advanced_synthetic_battery_data.csv"
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
    failure_log_file = "simulation_failures.csv"
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

plot_file = "pulse_discharge_plot.png"
plt.savefig(plot_file, dpi=150)
print(f"Plot saved to '{plot_file}'")

plt.show()
