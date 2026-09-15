import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

""" REPRODUCIBILITY & CONFIGURATION
Sets up an environment controls, random number seeds, and dynamic directory management.
"""
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Keep starting SOC >= 0.15 so continuous discharge does not trip the cutoff instantly at t=0
SOC_RANGE_MIN = float(os.environ.get("SOC_RANGE_MIN", 0.15))
SOC_RANGE_MAX = float(os.environ.get("SOC_RANGE_MAX", 1.0))

DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "continuous_discharge")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

OUTPUT_DATA_CSV = os.environ.get(
    "OUTPUT_DATA_CSV",
    os.path.join(RUN_DIR, "raw", "constant_current_synthetic_battery_data.csv"),
)
FAILURE_LOG_CSV = os.environ.get(
    "FAILURE_LOG_CSV",
    os.path.join(RUN_DIR, "failures", "simulation_failures.csv"),
)
DISCHARGE_PLOT_PNG = os.environ.get(
    "DISCHARGE_PLOT_PNG",
    os.path.join(RUN_DIR, "plots", "constant_current_discharge_plot.png"),
)

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

"""MODEL DEFINITION & PARAMETERS
Initializes the PyBaMM model, parameter sets, and helper functions for scaling and solving."""

# Standard isothermal SPM: robust across Prada2013 and NMC without missing thermal keys
model = pybamm.lithium_ion.SPM()

param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]

"""Lower voltage cutoff cant be 0.0V. Standard SPM does not iclude copper dissolution kinetics, 
i assumes intercalation never ceases until numerical failure.
=> will fail to reach true 0.0V.
Going to low generally harms calssifier robustness:
-The voltage bins below 2.0V will only capture a a couple of noisy points
    =>high column sparisty
-Below 2.0, both chemistries look nearly identical"""

LOWER_VOLTAGE_CUTOFF = {
    "LFP": 1.5,
    "NMC": 1.8,
}

"""Capacity scaling and experiment definition: Standardizes cell capacites
 so the ML classifier cannot use total capacity to infer chemistry"""
CAPACITY_TARGETS_AH = [1.2, 2.0, 3.5]

#base parameter sets have fixed nominal capacities
def capacity_multipliers_for(base_params, targets_ah):
    """Per-chemistry thickness multipliers scaling base parameters to target Ah."""
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    return [target_ah / base_capacity_ah for target_ah in targets_ah]

#scales the physical electrode thicknesses to achieve the target Ah for each chemistry
capacity_multipliers = {
    "LFP": capacity_multipliers_for(param_lfp_base, CAPACITY_TARGETS_AH),
    "NMC": capacity_multipliers_for(param_nmc_base, CAPACITY_TARGETS_AH),
}

#issues a single pybamm instruction to discharge at a fixed amapre until hitting V_min
def make_continuous_discharge_experiment(current_a, v_min):
    """Continuous constant-current discharge down to the voltage boundary."""
    return pybamm.Experiment([f"Discharge at {current_a:.4f} A until {v_min} V"])


FAILURE_LOG = []

"""SOLVER WRAPPER + ERROR HANDLING
Prevents indiviual simulation failures from crashing the entire sweep. 
Logs exceptions to a CSV for later inspection. 
Recovers any partial time-series data logged before termination"""
def solve_with_cutoff(sim, initial_soc, chem, context=""):
    """Solves the continuous discharge and logs exceptions gracefully."""
    try:
        sol = sim.solve(initial_soc=initial_soc)
    except pybamm.SolverError as err:
        sol = getattr(sim, "solution", None)
        recovered = sol is not None and len(sol.t) > 0
        print(
            f"  {chem}{context}: solver stopped early [SolverError] "
            f"({'partial data kept' if recovered else 'no data'}) ({err})"
        )
        FAILURE_LOG.append({
            "Chemistry": chem,
            "Context": context,
            "ExceptionType": "SolverError",
            "Message": str(err),
            "Recovered": recovered,
        })
        if not recovered:
            return None
    except Exception as err:
        print(f"  {chem}{context}: solve failed [{type(err).__name__}] ({err})")
        FAILURE_LOG.append({
            "Chemistry": chem,
            "Context": context,
            "ExceptionType": type(err).__name__,
            "Message": str(err),
            "Recovered": False,
        })
        return None

    termination = getattr(sol, "termination", None)
    if termination and termination != "final time":
        print(f"  {chem}: terminated early ({termination})")

    return sol

"""SIMUALTION SWEEP
Generates the synthetic dataset by varying operational and degradation parameters 
across 3x83=249 (498 total discharge curves)
1. Random sampling: Draws a distinct inital SOC, SOH, C-rate, and a random NMC paramter set (no not all three is used for each run)
2. Electrode degradation: Scales the maximum lithium concentration to simulate capacity fade/loss of active material (SOH)
3. Conductivity and thermal scaling: Alters electrode conductivity by temperature and aging factors to simulate internal resisitance growth
4. Execution and pairing: Discharges both LFP and NMC under identical conditions (I=c_rate*target_ah), logging the time-series data for each run.
5. Sensor noise injection: Adds Gaussian noise to the voltage signal to simulate real world sensor noise and measurement error.
6. Data assembly: Appends timestamps, voltage, capacity and metadata into tabular dataframes"""

variations_per_size = 83  # 3 sizes * 83 variations = ~249 runs per chemistry
all_data = []

print("Starting constant current-discharge simulations (Random C-rate/SOC/SOH/Temp)...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soc = random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX)
        soh = random.uniform(0.60, 0.85)

        # C-rate sampled per variation inside the loop (capped to SPM-safe range 0.2C-1.0C)
        c_rate = random.uniform(0.2, 1.0)

        ambient_c = random.uniform(0.0, 35.0)
        ambient_k = 273.15 + ambient_c

        variation_id = size_idx * variations_per_size + i
        nmc_set_name = random.choice(NMC_PARAMETER_SETS)

        print(
            f"\n--- Target: {target_ah} Ah | Var {i+1}/{variations_per_size} | "
            f"C-Rate: {c_rate:.2f}C | SOC: {soc:.2f} | SOH: {soh:.2f} | Temp: {ambient_c:.1f}C ---"
        )

        param_lfp = param_lfp_base.copy()
        param_nmc = param_nmc_bases[nmc_set_name].copy()

        chemistries = {
            "LFP": param_lfp,
            "NMC": param_nmc,
        }

        # Temperature-dependent conductivity factor
        temp_conductivity_multiplier = 1.0 + 0.02 * (ambient_c - 25.0)

        # Parameter updates per chemistry
        for chem, param in chemistries.items():
            mult = capacity_multipliers[chem][size_idx]

            # 1. Capacity scaling via electrode thicknesses
            param["Negative electrode thickness [m]"] *= mult
            param["Positive electrode thickness [m]"] *= mult

            # 2. Capacity loss (LAM/LLI) via max lithium concentration
            param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
            param["Maximum concentration in positive electrode [mol.m-3]"] *= soh

            # 3. Ambient temperature set where supported
            param["Ambient temperature [K]"] = ambient_k
            param["Initial temperature [K]"] = ambient_k

            # 4. Aging and temperature conductivity scaling
            resistance_noise = random.uniform(0.8, 1.2)
            conductivity_factor = min(1.0, max(0.3, soh * resistance_noise * temp_conductivity_multiplier))
            param["Negative electrode conductivity [S.m-1]"] *= conductivity_factor
            param["Positive electrode conductivity [S.m-1]"] *= conductivity_factor

        # Solve constant current discharge for both chemistries under identical conditions
        for chem, param in chemistries.items():
            v_min = LOWER_VOLTAGE_CUTOFF[chem]
            current_a = c_rate * target_ah
            discharge_experiment = make_continuous_discharge_experiment(current_a, v_min)
            sim = pybamm.Simulation(model, parameter_values=param, experiment=discharge_experiment)

            context = f" [target={target_ah}Ah, var={i+1}/{variations_per_size}, C-rate={c_rate:.2f}]"
            sol = solve_with_cutoff(sim, soc, chem, context=context)
            if sol is None:
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
                "C_Rate": round(c_rate, 3),
                "Current [A]": round(current_a, 4),
                "Size_Multiplier": capacity_multipliers[chem][size_idx],
                "SOH": round(soh, 3),
                "Initial_SOC": round(soc, 3),
                "Ambient_Temperature_C": round(ambient_c, 2),
                "Base_Parameter_Set": nmc_set_name if chem == "NMC" else "Prada2013",
                "V_min [V]": v_min,
                "Variation_ID": variation_id,
            })
            all_data.append(df)

"""SAVE DATA
Merges individual time series dataframes into a single master CSV.
Saves the raw dataset to disk for the feature engineering script and 
exports simulation failures if any runs failed"""

if all_data:
    training_data = pd.concat(all_data, ignore_index=True)
    training_data.to_csv(OUTPUT_DATA_CSV, index=False)
    print(f"\nDone! Successfully saved data to '{OUTPUT_DATA_CSV}'")

total_attempts = len(CAPACITY_TARGETS_AH) * variations_per_size * len(LOWER_VOLTAGE_CUTOFF)
print(f"\n{len(FAILURE_LOG)} of {total_attempts} solve attempts failed.")
if FAILURE_LOG:
    failures_df = pd.DataFrame(FAILURE_LOG)
    failures_df.to_csv(FAILURE_LOG_CSV, index=False)
    print(failures_df["ExceptionType"].value_counts().to_string())
    print(f"Full failure log saved to '{FAILURE_LOG_CSV}'")

"""PLOT ALL RUNS
Generates a dual panel visulaization to inspect dataset quality. 
Left plot: Overlays all LPF and NMC curves to visually verify the flat LFP plateau
against the sloping NMC curve across all cell sizes, C-rates, and SOC/SOH variations.
Right plot: Normalizes the x-axis to relative capacity (Ah / Ah_nominal) and color codes curves by C-rate.
This reveals the ohmic (IR) drops that the ML pipleine must navigate to classify"""

print("Generatingdischarge plot containing all runs...")

if not training_data.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # 1. Plot Left: Chemistries side-by-side / overlayed
    lfp_legend_added = False
    nmc_legend_added = False

    for (chem, var_id), run_df in training_data.groupby(["Chemistry", "Variation_ID"]):
        if chem == "LFP":
            ax1.plot(
                run_df["Capacity [A.h]"],
                run_df["Voltage [V]"],
                color="#1f77b4",
                alpha=0.18,
                linewidth=1.0,
                label="LFP Runs" if not lfp_legend_added else "",
            )
            lfp_legend_added = True
        else:
            ax1.plot(
                run_df["Capacity [A.h]"],
                run_df["Voltage [V]"],
                color="#ff7f0e",
                alpha=0.18,
                linewidth=1.0,
                label="NMC Runs" if not nmc_legend_added else "",
            )
            nmc_legend_added = True

    ax1.set_title("All Discharges: LFP vs. NMC", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Discharged Capacity [Ah]", fontsize=11)
    ax1.set_ylabel("Terminal Voltage [V]", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Re-draw bold handles for the legend so lines are clearly visible
    leg1 = ax1.legend(loc="upper right", frameon=True)
    for line in leg1.get_lines():
        line.set_alpha(1.0)
        line.set_linewidth(2.5)

    # 2. Plot Right: Normalized by Depth of Discharge, Colored by C-Rate
    # Normalizing capacity by each run's target capacity reveals the intrinsic OCV plateau
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.2, vmax=3.0)

    for (chem, var_id), run_df in training_data.groupby(["Chemistry", "Variation_ID"]):
        c_rate = run_df["C_Rate"].iloc[0]
        color = cmap(norm(c_rate))
        linestyle = "-" if chem == "LFP" else "--"
        
        # Approximate relative discharge window (Ah / Target Ah)
        rel_cap = run_df["Capacity [A.h]"] / run_df["Target_Capacity_Ah"].iloc[0]

        ax2.plot(
            rel_cap,
            run_df["Voltage [V]"],
            color=color,
            linestyle=linestyle,
            alpha=0.22,
            linewidth=1.0,
        )

    ax2.set_title("Color-Coded by C-Rate (Solid: LFP, Dashed: NMC)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Relative Discharged Capacity [Ah / Ah_nominal]", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Add Colorbar for C-rates
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, pad=0.02)
    cbar.set_label("Discharge C-Rate [C]", fontsize=10)
# At the end of the simulation plotting block:
    plt.tight_layout()
    plt.savefig(DISCHARGE_PLOT_PNG, dpi=200)
    print(f"All-run overview plot successfully saved to '{DISCHARGE_PLOT_PNG}'")
    plt.close()
else:
    print("Plot skipped: training_data is empty.")