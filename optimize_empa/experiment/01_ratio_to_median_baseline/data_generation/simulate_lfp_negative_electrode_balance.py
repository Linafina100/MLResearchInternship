"""
optimize_empa, attempt 01 (copied in from experiments/32_fixed_voltage_cutoff_soc_sweep/
for self-containment -- see this folder's RESULTS.md for the full
derivation history): regenerates ONLY the LFP side of the continuous-
discharge-truncated synthetic set, with the negative-electrode capacity-
balance fix (factor=0.7 shifts LFP's dV/dQ dive to ~2.67V, close to real
EMPA's ~2.65V) applied. NOT invoked by run_evaluation.py -- kept here so
this experiment's synthetic data is fully reproducible from local files
alone, without depending on the old experiments/32_ lineage.

RESOURCEFUL REUSE: NMC is NOT touched by this fix at all, so this script
reproduces the EXACT same per-variation (SOH, C-rate, ambient
temperature) draws as the local
simulate_batteries_continuous_discharge_truncated.py (same seed=42, same
loop structure -- including the unused NMC parameter-set draw, kept only
to stay in sync with the original random stream) but only SOLVES the LFP
side. The output is combined with that script's unmodified NMC rows by
the local combine_lfp_fix_with_baseline_nmc.py -- no NMC resimulation
needed.

Usage: python3 optimize_empa/experiment/01_ratio_to_median_baseline/data_generation/simulate_lfp_negative_electrode_balance.py
"""
import os
import sys
import pybamm
import pandas as pd
import numpy as np
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from soc_truncation import truncate_at_soc

random.seed(42)
np.random.seed(42)

INITIAL_SOC_FULL = 0.99
SOH_MIN, SOH_MAX = 0.8, 1.0
C_RATE_MIN, C_RATE_MAX = 0.1, 0.2
LFP_OCP_RATE_CONSTANT = -3
NEG_CAPACITY_FACTOR = float(os.environ.get("NEG_CAPACITY_FACTOR", 0.7))

SOC_TRUNCATION_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

DATA_DIR = "data"
RUN_LABEL = os.environ.get("RUN_LABEL", "optimize_empa_01_lfp_negative_electrode_balance")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)
OUTPUT_DATA_CSV = os.path.join(RUN_DIR, "raw", "lfp_only_balance_fix.csv")
os.makedirs(os.path.dirname(OUTPUT_DATA_CSV), exist_ok=True)

model = pybamm.lithium_ion.SPM()
param_lfp_base = pybamm.ParameterValues("Prada2013")
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in ["Chen2020", "OKane2022"]}  # unused, kept only for RNG sync

LOWER_VOLTAGE_CUTOFF_LFP = 1.5
CAPACITY_TARGETS_AH = [1.2, 2.0, 3.5]


def capacity_multipliers_for(base_params, targets_ah):
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    return [target_ah / base_capacity_ah for target_ah in targets_ah]


capacity_multipliers_lfp = capacity_multipliers_for(param_lfp_base, CAPACITY_TARGETS_AH)


def make_lfp_ocp(rate_constant):
    def lfp_ocp_softened(sto):
        c1 = -150 * sto
        c2 = rate_constant * (1 - sto)
        k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)
        return k
    return lfp_ocp_softened


def make_continuous_discharge_experiment(current_a, v_min):
    return pybamm.Experiment([f"Discharge at {current_a:.4f} A until {v_min} V"])


T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6


def solve_dense(param, discharge_experiment, initial_soc, context=""):
    sim1 = pybamm.Simulation(model, parameter_values=param, experiment=discharge_experiment)
    try:
        sol1 = sim1.solve(initial_soc=initial_soc)
    except Exception as err:
        print(f"  LFP{context}: solve failed ({type(err).__name__}: {err})")
        return None

    time_default = sol1["Time [s]"].entries
    voltage_default = sol1["Terminal voltage [V]"].entries
    capacity_default = sol1["Discharge capacity [A.h]"].entries
    if len(time_default) < 5:
        return time_default, voltage_default, capacity_default

    tf = time_default[-1]
    t_bound = tf * T_INTERP_SAFETY_FRACTION
    try:
        sim2 = pybamm.Simulation(model, parameter_values=param, experiment=discharge_experiment)
        t_interp = np.linspace(0, t_bound, T_INTERP_N_POINTS)
        sol2 = sim2.solve(initial_soc=initial_soc, t_interp=t_interp)
    except Exception as err:
        print(f"  LFP{context}: t_interp pass failed ({type(err).__name__}: {err}) -- using default density.")
        return time_default, voltage_default, capacity_default

    time_dense = sol2["Time [s]"].entries
    voltage_dense = sol2["Terminal voltage [V]"].entries
    capacity_dense = sol2["Discharge capacity [A.h]"].entries
    after_mask = time_default > time_dense[-1]
    time_spliced = np.concatenate([time_dense, time_default[after_mask]])
    voltage_spliced = np.concatenate([voltage_dense, voltage_default[after_mask]])
    capacity_spliced = np.concatenate([capacity_dense, capacity_default[after_mask]])
    order = np.argsort(time_spliced, kind="stable")
    return time_spliced[order], voltage_spliced[order], capacity_spliced[order]


variations_per_size = int(os.environ.get("VARIATIONS_PER_SIZE", 83))  # 83 matches the original baseline exactly
all_data = []
n_failed = 0

print(f"Regenerating LFP ONLY with negative-electrode capacity factor={NEG_CAPACITY_FACTOR} "
      f"(NMC untouched -- reused from the existing baseline CSV). Same seed=42 as the original "
      f"generator, so SOH/C-rate/temperature draws per variation match exactly.")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soh = random.uniform(SOH_MIN, SOH_MAX)
        c_rate = random.uniform(C_RATE_MIN, C_RATE_MAX)
        ambient_c = random.uniform(15.0, 35.0)
        TEMP_RESISTANCE_COEFF = 0.02
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)
        variation_id = size_idx * variations_per_size + i
        _ = random.choice(["Chen2020", "OKane2022"])  # unused NMC draw, kept for RNG sync with the original

        param = param_lfp_base.copy()
        mult = capacity_multipliers_lfp[size_idx]
        param["Negative electrode thickness [m]"] *= mult
        param["Positive electrode thickness [m]"] *= mult

        param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
        param["Maximum concentration in positive electrode [mol.m-3]"] *= soh
        param["Initial concentration in negative electrode [mol.m-3]"] *= soh
        param["Initial concentration in positive electrode [mol.m-3]"] *= soh

        # THE FIX under test
        param["Maximum concentration in negative electrode [mol.m-3]"] *= NEG_CAPACITY_FACTOR
        param["Initial concentration in negative electrode [mol.m-3]"] *= NEG_CAPACITY_FACTOR

        resistance_noise = random.uniform(0.8, 1.2)
        resistance_factor = min(1.0, max(0.3, soh * resistance_noise * temp_conductivity_multiplier))
        param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
        param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

        param["Positive electrode OCP [V]"] = make_lfp_ocp(LFP_OCP_RATE_CONSTANT)

        v_min = LOWER_VOLTAGE_CUTOFF_LFP
        current_a = c_rate * target_ah
        discharge_experiment = make_continuous_discharge_experiment(current_a, v_min)
        context = f" [target={target_ah}Ah, variation={i+1}/{variations_per_size}, SOH={soh:.3f}, C-rate={c_rate:.3f}]"
        print(f"Solving LFP {context}...")
        dense = solve_dense(param, discharge_experiment, INITIAL_SOC_FULL, context=context)
        if dense is None:
            n_failed += 1
            continue
        time_arr, voltage_arr, capacity_arr = dense
        noise = np.random.normal(0, 0.001, len(voltage_arr))
        voltage_with_noise = voltage_arr + noise

        base_row = {
            "Chemistry": "LFP", "Target_Capacity_Ah": target_ah, "C_Rate": round(c_rate, 4),
            "Size_Multiplier": capacity_multipliers_lfp[size_idx], "SOH": round(soh, 3),
            "Ambient_Temperature_C": round(ambient_c, 2), "Resistance_Factor": round(resistance_factor, 4),
            "Base_Parameter_Set": "Prada2013", "V_min [V]": v_min,
            "Neg_Capacity_Factor": NEG_CAPACITY_FACTOR,
        }

        for s in SOC_TRUNCATION_POINTS:
            truncated = truncate_at_soc(time_arr, voltage_with_noise, capacity_arr, s)
            if truncated is None:
                continue
            t_trunc, v_trunc, cap_trunc = truncated
            df = pd.DataFrame({
                "Time [s]": t_trunc, "Voltage [V]": v_trunc, "Capacity [A.h]": cap_trunc,
                "Initial_SOC": s, "Variation_ID": f"{variation_id}_soc_{s}", **base_row,
            })
            all_data.append(df)

training_data = pd.concat(all_data, ignore_index=True)
training_data.to_csv(OUTPUT_DATA_CSV, index=False)
print(f"\nDone! {n_failed} failed solves. Saved to '{OUTPUT_DATA_CSV}' "
      f"({training_data['Variation_ID'].str.split('_soc_').str[0].nunique()} base configs)")
