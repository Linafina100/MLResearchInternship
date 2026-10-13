"""
Experiment 32: generates ADDITIONAL continuous-discharge-truncated
synthetic battery variations -- same physics, same calibration, same
method as experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/simulate_batteries_continuous_discharge_truncated.py
(read that script's docstring for the full method) -- to be concatenated
with the EXISTING data/26_soh_range_continuous_discharge_truncated_v1/
dataset (249 battery configs per chemistry, Variation_ID 0-248) rather
than regenerating it from scratch.

WHY: experiment 32's per-SOC-specialized models (one Random Forest per
Initial_SOC point, not pooled) reproduced experiment 24's near-perfect
EMPA numbers, but confirmed exp07 needs MORE synthetic training examples
per SOC point than the 249-per-chemistry base set provides -- isolated
directly: training on only the Initial_SOC=1.0 slice (498 total
examples) crashes exp07 NMC recall from 0.964 (full 5,976-example pooled
set) to 0.428, with bin selection and real test data held identical. The
fix tested here: brute-force 10x the number of base battery configs
(249 -> ~2,490 per chemistry) so each per-SOC-point model has ~10x the
training examples, without changing anything about the per-SOC-point
methodology itself.

RESOURCEFUL REUSE: rather than resimulating the original 249 configs,
this script only generates the INCREMENTAL ~9x of NEW configs, using a
DIFFERENT random seed (so they're genuinely new draws, not a repeat of
the original stream) and a Variation_ID range offset far above the
original's 0-248 (default 10000+) so the two CSVs can be concatenated
with zero ID collisions. See combine_synthetic_data.py in this folder
for the concatenation step.

Otherwise byte-for-byte identical to experiment 26's script: SOH 0.8-1.0
range, C-rate 0.1-0.2, NMC diffusivity/10 fix, LFP OCP rate=-3 fix,
1.5V natural cutoffs, dense t_interp splicing.

Usage: env vars RANDOM_SEED (default 9001, must differ from the
original's 42), VARIATION_ID_OFFSET (default 10000), VARIATIONS_PER_SIZE
(default 747, i.e. 9x the original 83, for a combined 830 = 10x total),
DATA_DIR, RUN_LABEL, OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG,
SOH_MIN/MAX, C_RATE_MIN/MAX, DIFFUSIVITY_FACTOR, LFP_OCP_RATE_CONSTANT
(all optional).
"""
import os
import sys
import pybamm
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXP26_DIR = os.path.join(PROJECT_DIR, "experiments", "22_29_sim_to_real_validation", "26_soc_sweep_three_datasets")
sys.path.insert(0, EXP26_DIR)
from soc_truncation import truncate_at_soc

RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 9001))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

VARIATION_ID_OFFSET = int(os.environ.get("VARIATION_ID_OFFSET", 10000))

INITIAL_SOC_FULL = float(os.environ.get("INITIAL_SOC_FULL", 0.99))
SOH_MIN = float(os.environ.get("SOH_MIN", 0.8))
SOH_MAX = float(os.environ.get("SOH_MAX", 1.0))
C_RATE_MIN = float(os.environ.get("C_RATE_MIN", 0.1))
C_RATE_MAX = float(os.environ.get("C_RATE_MAX", 0.2))
DIFFUSIVITY_FACTOR = float(os.environ.get("DIFFUSIVITY_FACTOR", 10))
LFP_OCP_RATE_CONSTANT = float(os.environ.get("LFP_OCP_RATE_CONSTANT", -3))

SOC_TRUNCATION_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "32_synthetic_extra_variations")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data_extra.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
DISCHARGE_PLOT_PNG = os.environ.get("DISCHARGE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "continuous_discharge_plot.png"))

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

model = pybamm.lithium_ion.SPM()

param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]

LOWER_VOLTAGE_CUTOFF = {
    "LFP": float(os.environ.get("LFP_LOWER_CUTOFF", 1.5)),
    "NMC": float(os.environ.get("NMC_LOWER_CUTOFF", 1.5)),
}


def apply_nmc_diffusivity_fix(param, factor):
    orig = param["Positive particle diffusivity [m2.s-1]"]
    if callable(orig):
        def scaled_diffusivity(sto, T, _orig=orig, _factor=factor):
            return _orig(sto, T) / _factor
        param["Positive particle diffusivity [m2.s-1]"] = scaled_diffusivity
    else:
        param["Positive particle diffusivity [m2.s-1]"] = orig / factor
    return param


def make_lfp_ocp(rate_constant):
    def lfp_ocp_softened(sto):
        c1 = -150 * sto
        c2 = rate_constant * (1 - sto)
        k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)
        return k
    return lfp_ocp_softened


def apply_lfp_ocp_fix(param, rate_constant):
    param["Positive electrode OCP [V]"] = make_lfp_ocp(rate_constant)
    return param


def make_continuous_discharge_experiment(current_a, v_min):
    return pybamm.Experiment([f"Discharge at {current_a:.4f} A until {v_min} V"])


FAILURE_LOG = []


def solve_with_cutoff(sim, initial_soc, chem, context=""):
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


T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6


def solve_dense(param, discharge_experiment, initial_soc, chem, context=""):
    sim1 = pybamm.Simulation(model, parameter_values=param, experiment=discharge_experiment)
    sol1 = solve_with_cutoff(sim1, initial_soc, chem, context=context)
    if sol1 is None:
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
        print(f"  {chem}{context}: t_interp dense pass failed ({type(err).__name__}: {err}) -- "
              f"falling back to default-density output for this battery.")
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


CAPACITY_TARGETS_AH = [1.2, 2.0, 3.5]


def capacity_multipliers_for(base_params, targets_ah):
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    return [target_ah / base_capacity_ah for target_ah in targets_ah]


capacity_multipliers = {
    "LFP": capacity_multipliers_for(param_lfp_base, CAPACITY_TARGETS_AH),
    "NMC": capacity_multipliers_for(param_nmc_base, CAPACITY_TARGETS_AH),
}

variations_per_size = int(os.environ.get("VARIATIONS_PER_SIZE", 747))

all_data = []
full_traces_for_plot = []

print(f"Starting EXTRA continuous-discharge simulations (seed={RANDOM_SEED}, "
      f"Variation_ID offset={VARIATION_ID_OFFSET}, {variations_per_size}/size, "
      f"SOH sampled {SOH_MIN}-{SOH_MAX}, C-rate {C_RATE_MIN}-{C_RATE_MAX}, "
      f"cutoffs LFP={LOWER_VOLTAGE_CUTOFF['LFP']}V/NMC={LOWER_VOLTAGE_CUTOFF['NMC']}V, "
      f"NMC parameter sets {NMC_PARAMETER_SETS} with diffusivity/{DIFFUSIVITY_FACTOR}, "
      f"LFP OCP tail rate constant {LFP_OCP_RATE_CONSTANT}), to be concatenated with "
      f"the existing 249-per-chemistry base set...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soh = random.uniform(SOH_MIN, SOH_MAX)
        c_rate = random.uniform(C_RATE_MIN, C_RATE_MAX)

        ambient_c = random.uniform(15.0, 35.0)
        TEMP_RESISTANCE_COEFF = 0.02
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)

        variation_id = VARIATION_ID_OFFSET + size_idx * variations_per_size + i

        nmc_set_name = random.choice(NMC_PARAMETER_SETS)

        print(f"\n--- Target size: {target_ah} Ah | Variation {i+1}/{variations_per_size} | "
              f"C-rate: {c_rate:.3f} | SOH: {soh:.3f} | Ambient: {ambient_c:.1f}C | NMC set: {nmc_set_name} ---")

        param_lfp = param_lfp_base.copy()
        param_lfp = apply_lfp_ocp_fix(param_lfp, LFP_OCP_RATE_CONSTANT)
        param_nmc = param_nmc_bases[nmc_set_name].copy()
        param_nmc = apply_nmc_diffusivity_fix(param_nmc, DIFFUSIVITY_FACTOR)

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
            param["Initial concentration in negative electrode [mol.m-3]"] *= soh
            param["Initial concentration in positive electrode [mol.m-3]"] *= soh

            resistance_noise = random.uniform(0.8, 1.2)
            resistance_factor = min(1.0, max(0.3, soh * resistance_noise * temp_conductivity_multiplier))
            resistance_factors[chem] = resistance_factor
            param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
            param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

        for chem, param in chemistries.items():
            v_min = LOWER_VOLTAGE_CUTOFF[chem]
            current_a = c_rate * target_ah
            discharge_experiment = make_continuous_discharge_experiment(current_a, v_min)

            print(f"Solving {chem} (I={current_a:.4f} A / {c_rate:.3f}C, Vmin={v_min} V, dense t_interp output)...")
            context = f" [target={target_ah}Ah, variation={i+1}/{variations_per_size}, SOH={soh:.3f}, C-rate={c_rate:.3f}]"
            dense = solve_dense(param, discharge_experiment, INITIAL_SOC_FULL, chem, context=context)
            if dense is None:
                print(f"  Skipping {chem}: no solution data.")
                continue
            time_arr, voltage_arr, capacity_arr = dense

            raw_voltage = voltage_arr
            noise = np.random.normal(0, 0.001, len(raw_voltage))
            voltage_with_noise = raw_voltage + noise

            full_traces_for_plot.append(pd.DataFrame({
                "Time [s]": time_arr, "Voltage [V]": voltage_with_noise,
                "Chemistry": chem, "Variation_ID": variation_id,
            }))

            base_row = {
                "Chemistry": chem,
                "Target_Capacity_Ah": target_ah,
                "C_Rate": round(c_rate, 4),
                "Size_Multiplier": capacity_multipliers[chem][size_idx],
                "SOH": round(soh, 3),
                "Ambient_Temperature_C": round(ambient_c, 2),
                "Resistance_Factor": round(resistance_factors[chem], 4),
                "Base_Parameter_Set": nmc_set_name if chem == "NMC" else "Prada2013",
                "V_min [V]": v_min,
            }

            for s in SOC_TRUNCATION_POINTS:
                truncated = truncate_at_soc(time_arr, voltage_with_noise, capacity_arr, s)
                if truncated is None:
                    print(f"  {chem} Initial_SOC={s}: too few points after truncation, skipped.")
                    continue
                t_trunc, v_trunc, cap_trunc = truncated
                df = pd.DataFrame({
                    "Time [s]": t_trunc,
                    "Voltage [V]": v_trunc,
                    "Capacity [A.h]": cap_trunc,
                    "Initial_SOC": s,
                    "Variation_ID": f"{variation_id}_soc_{s}",
                    **base_row,
                })
                all_data.append(df)

if all_data:
    training_data = pd.concat(all_data, ignore_index=True)
    training_data.to_csv(OUTPUT_DATA_CSV, index=False)
    print(f"\nDone! Extra continuous-discharge-truncated data saved to '{OUTPUT_DATA_CSV}'")
    print(f"  {training_data['Variation_ID'].nunique()} battery/SOC-point rows "
          f"from {training_data.groupby('Chemistry')['Variation_ID'].apply(lambda s: s.str.split('_soc_').str[0].nunique()).to_dict()} base full-discharge traces per chemistry")
else:
    training_data = pd.DataFrame()
    print("\nNo data generated: every solve attempt failed.")

total_attempts = len(CAPACITY_TARGETS_AH) * variations_per_size * len(LOWER_VOLTAGE_CUTOFF)
print(f"\n{len(FAILURE_LOG)} of {total_attempts} solve attempts failed.")
if FAILURE_LOG:
    failures_df = pd.DataFrame(FAILURE_LOG)
    print(failures_df["ExceptionType"].value_counts().to_string())
    failures_df.to_csv(FAILURE_LOG_CSV, index=False)
    print(f"Full failure log saved to '{FAILURE_LOG_CSV}'")

if full_traces_for_plot:
    print("Generating continuous discharge plot (full traces, pre-truncation)...")
    plot_df = pd.concat(full_traces_for_plot, ignore_index=True)
    plt.figure(figsize=(12, 6))
    lfp_labeled, nmc_labeled = False, False
    for (chem, var_id), run_df in plot_df.groupby(["Chemistry", "Variation_ID"]):
        color = '#1f77b4' if chem == "LFP" else '#ff7f0e'
        label = None
        if chem == "LFP" and not lfp_labeled:
            label, lfp_labeled = "LFP", True
        elif chem == "NMC" and not nmc_labeled:
            label, nmc_labeled = "NMC", True
        plt.plot(run_df['Time [s]'] / 3600, run_df['Voltage [V]'], color=color, alpha=0.08, linewidth=0.6, label=label)

    plt.title(f'Extra Full Discharge Profiles Pre-Truncation (seed={RANDOM_SEED}, SOH {SOH_MIN}-{SOH_MAX})')
    plt.xlabel('Time [Hours]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid(True)

    plt.savefig(DISCHARGE_PLOT_PNG, dpi=150)
    print(f"Plot saved to '{DISCHARGE_PLOT_PNG}'")
    plt.close()
else:
    print("Plot skipped: no data generated.")
