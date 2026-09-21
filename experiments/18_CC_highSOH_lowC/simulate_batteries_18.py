import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

""" REPRODUCIBILITY & CONFIGURATION
Sets up environment controls, random number seeds, and dynamic directory management. 
"""
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SOC_RANGE_MIN = float(os.environ.get("SOC_RANGE_MIN", 0.05))
SOC_RANGE_MAX = float(os.environ.get("SOC_RANGE_MAX", 1.0))

DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "continuous_discharge")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

OUTPUT_DATA_CSV = os.environ.get(
    "OUTPUT_DATA_CSV",
    os.path.join(RUN_DIR, "raw", "constant_current_synthetic_battery_data_18.csv"),
)
FAILURE_LOG_CSV = os.environ.get(
    "FAILURE_LOG_CSV",
    os.path.join(RUN_DIR, "failures", "simulation_failures_18.csv"),
)
DISCHARGE_PLOT_PNG = os.environ.get(
    "DISCHARGE_PLOT_PNG",
    os.path.join(RUN_DIR, "plots", "constant_current_discharge_plot_18.png"),
)

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

"""MODEL DEFINITION & PARAMETERS"""
model = pybamm.lithium_ion.SPM()

param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]

LOWER_VOLTAGE_CUTOFF = {
    "LFP": 1.5,
    "NMC": 1.8,
}

CAPACITY_TARGETS_AH = [1.2, 2.0, 3.5]

def capacity_multipliers_for(base_params, targets_ah):
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    return [target_ah / base_capacity_ah for target_ah in targets_ah]

capacity_multipliers = {
    "LFP": capacity_multipliers_for(param_lfp_base, CAPACITY_TARGETS_AH),
    "NMC": capacity_multipliers_for(param_nmc_base, CAPACITY_TARGETS_AH),
}

def make_continuous_discharge_experiment(current_a, v_min):
    return pybamm.Experiment([f"Discharge at {current_a:.4f} A until {v_min} V"])


FAILURE_LOG = []

def solve_with_cutoff(sim, initial_soc, chem, context=""):
    try:
        sol = sim.solve(initial_soc=initial_soc)
    except pybamm.SolverError as err:
        sol = getattr(sim, "solution", None)
        recovered = sol is not None and len(sol.t) > 0
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
        FAILURE_LOG.append({
            "Chemistry": chem,
            "Context": context,
            "ExceptionType": type(err).__name__,
            "Message": str(err),
            "Recovered": False,
        })
        return None
    return sol

"""SIMULATION SWEEP: DETERMINISTIC NESTED GRID (3 PREDEFINED C-RATES)"""
PREDEFINED_C_RATES = [0.1, 0.2] #changed from 0.2, 0.5, 1
variations_per_size = 40  # 3 sizes * 40 variations * 2 rates = 240 runs per chemistry 
all_data = []
var_counter = 0

print(f"Starting constant current simulations with discrete C-rates {PREDEFINED_C_RATES}...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soc = random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX)
        soh = random.uniform(0.80, 0.95) #changed from 0.5,0.8
        ambient_c = random.uniform(15.0, 35.0) #chnged from 0.0, 35.0
        ambient_k = 273.15 + ambient_c
        nmc_set_name = random.choice(NMC_PARAMETER_SETS)

        for c_rate in PREDEFINED_C_RATES:
            variation_id = var_counter
            var_counter += 1

            print(
                f"\n--- Target: {target_ah} Ah | Base Var {i+1}/{variations_per_size} | "
                f"C-Rate: {c_rate}C | SOC: {soc:.2f} | SOH: {soh:.2f} | Temp: {ambient_c:.1f}C ---"
            )

            param_lfp = param_lfp_base.copy()
            param_nmc = param_nmc_bases[nmc_set_name].copy()
            chemistries = {"LFP": param_lfp, "NMC": param_nmc}

            temp_conductivity_multiplier = 1.0 + 0.02 * (ambient_c - 25.0)

            for chem, param in chemistries.items():
                mult = capacity_multipliers[chem][size_idx]
                param["Negative electrode thickness [m]"] *= mult
                param["Positive electrode thickness [m]"] *= mult
                param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
                param["Maximum concentration in positive electrode [mol.m-3]"] *= soh
                param["Ambient temperature [K]"] = ambient_k
                param["Initial temperature [K]"] = ambient_k

                resistance_noise = random.uniform(0.8, 1.2)
                conductivity_factor = min(1.0, max(0.3, soh * resistance_noise * temp_conductivity_multiplier))
                param["Negative electrode conductivity [S.m-1]"] *= conductivity_factor
                param["Positive electrode conductivity [S.m-1]"] *= conductivity_factor

            for chem, param in chemistries.items():
                v_min = LOWER_VOLTAGE_CUTOFF[chem]
                current_a = c_rate * target_ah
                discharge_experiment = make_continuous_discharge_experiment(current_a, v_min)
                sim = pybamm.Simulation(model, parameter_values=param, experiment=discharge_experiment)

                context = f" [target={target_ah}Ah, var={variation_id}, C-rate={c_rate}C]"
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

if all_data:
    training_data = pd.concat(all_data, ignore_index=True)
    training_data.to_csv(OUTPUT_DATA_CSV, index=False)
    print(f"\nDone! Successfully saved data to '{OUTPUT_DATA_CSV}'")
else:
    training_data = pd.DataFrame()

total_attempts = len(CAPACITY_TARGETS_AH) * variations_per_size * len(PREDEFINED_C_RATES) * len(LOWER_VOLTAGE_CUTOFF)
print(f"\n{len(FAILURE_LOG)} of {total_attempts} solve attempts failed.")
if FAILURE_LOG:
    failures_df = pd.DataFrame(FAILURE_LOG)
    failures_df.to_csv(FAILURE_LOG_CSV, index=False)

"""PLOT ALL RUNS"""
print("Generating discharge plot containing all runs...")

if not training_data.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    lfp_legend_added = False
    nmc_legend_added = False

    for (chem, var_id), run_df in training_data.groupby(["Chemistry", "Variation_ID"]):
        if chem == "LFP":
            ax1.plot(run_df["Capacity [A.h]"], run_df["Voltage [V]"], color="#1f77b4", alpha=0.18, linewidth=1.0, label="LFP Runs" if not lfp_legend_added else "")
            lfp_legend_added = True
        else:
            ax1.plot(run_df["Capacity [A.h]"], run_df["Voltage [V]"], color="#ff7f0e", alpha=0.18, linewidth=1.0, label="NMC Runs" if not nmc_legend_added else "")
            nmc_legend_added = True

    ax1.set_title("All Discharges: LFP vs. NMC", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Discharged Capacity [Ah]", fontsize=11)
    ax1.set_ylabel("Terminal Voltage [V]", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    leg1 = ax1.legend(loc="upper right", frameon=True)
    for line in leg1.get_lines():
        line.set_alpha(1.0)
        line.set_linewidth(2.5)

    color_map = {0.1: "#2ca02c", 0.2: "#d62728"}

    for (chem, var_id), run_df in training_data.groupby(["Chemistry", "Variation_ID"]):
        c_rate = run_df["C_Rate"].iloc[0]
        color = color_map.get(c_rate, "#7f7f7f")
        linestyle = "-" if chem == "LFP" else "--"
        rel_cap = run_df["Capacity [A.h]"] / run_df["Target_Capacity_Ah"].iloc[0]

        ax2.plot(rel_cap, run_df["Voltage [V]"], color=color, linestyle=linestyle, alpha=0.22, linewidth=1.0)

    ax2.set_title("Discrete C-Rates (Solid: LFP, Dashed: NMC)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Relative Discharged Capacity [Ah / Ah_nominal]", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    handles = [plt.Line2D([0], [0], color=color_map[rate], lw=2, label=f"{rate}C") for rate in PREDEFINED_C_RATES]
    ax2.legend(handles=handles, title="C-Rate", loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(DISCHARGE_PLOT_PNG, dpi=200)
    print(f"All-run overview plot successfully saved to '{DISCHARGE_PLOT_PNG}'")
    plt.close(fig)