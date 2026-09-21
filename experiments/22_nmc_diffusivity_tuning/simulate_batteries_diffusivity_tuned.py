"""
Experiment 22, Phase 2: applies the diffusivity-tuning fix found in this
experiment's Phase 1 diagnostic (diagnose_diffusivity_factor.py) to
experiment 20's low-voltage-only, SOH=0.8 sim-to-real pipeline, on top
of the SOH-scaling fix (experiment 20) and t_interp dense sampling
(experiment 18) -- targeting the 3-6x dV/dQ magnitude mismatch
experiment 20 exposed once coverage was no longer masking it (experiment
21 confirmed post-hoc feature normalization can't fix this; it has to be
fixed at the simulation source).

Fix: `Positive particle diffusivity [m2.s-1]` (NMC's own active-material
parameter) reduced 10x. Phase 1 found this is a real, mechanistically-
sound lever (slower particle-surface concentration tracking smears the
end-of-discharge voltage transition) that works cleanly for Chen2020 and
OKane2022 across the full realistic C-rate (0.1-0.2)/temperature
(15-35C) range at SOH=0.8 (target-zone dV/dQ magnitude moves from ~-8/-10
to ~-6, landing inside real NMC's -8 to -3 range, zero outliers) -- but
does essentially nothing for Mohtat2020 (stays ~-26 to -28 regardless of
factor). Per explicit user decision, **Mohtat2020 is dropped from the
NMC parameter-set pool for this experiment** rather than diluting the
result or guessing a separate fix for it; it remains open for future
work.

Otherwise identical to experiment 20: SOH fixed at 0.8, C-rate
randomized 0.1-0.2, ambient temperature randomized 15-35C, 1.5V cutoff
both chemistries, dense t_interp output for every battery.

Usage: env vars DATA_DIR, RUN_LABEL, OUTPUT_DATA_CSV, FAILURE_LOG_CSV,
DISCHARGE_PLOT_PNG, LFP_LOWER_CUTOFF, NMC_LOWER_CUTOFF, SOH_FIXED,
C_RATE_MIN/MAX, SOC_RANGE_MIN/MAX, DIFFUSIVITY_FACTOR (all optional).
"""
import os
import pybamm
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

SOC_RANGE_MIN = float(os.environ.get("SOC_RANGE_MIN", 0.5))
SOC_RANGE_MAX = float(os.environ.get("SOC_RANGE_MAX", 1.0))
SOH_FIXED = float(os.environ.get("SOH_FIXED", 0.8))
C_RATE_MIN = float(os.environ.get("C_RATE_MIN", 0.1))
C_RATE_MAX = float(os.environ.get("C_RATE_MAX", 0.2))
DIFFUSIVITY_FACTOR = float(os.environ.get("DIFFUSIVITY_FACTOR", 10))

DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "soh_0.8_diffusivity_tuned_v1.5")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
DISCHARGE_PLOT_PNG = os.environ.get("DISCHARGE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "continuous_discharge_plot.png"))

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

model = pybamm.lithium_ion.SPM()

param_lfp_base = pybamm.ParameterValues("Prada2013")
# Mohtat2020 dropped: Phase 1 found diffusivity tuning doesn't fix its
# magnitude mismatch at all (stays ~-26 to -28 regardless of factor),
# unlike Chen2020/OKane2022 -- see RESULTS.md. Not diluting this
# experiment's result with an unfixed third of the pool, per explicit
# decision.
NMC_PARAMETER_SETS = ["Chen2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]

LOWER_VOLTAGE_CUTOFF = {
    "LFP": float(os.environ.get("LFP_LOWER_CUTOFF", 1.5)),
    "NMC": float(os.environ.get("NMC_LOWER_CUTOFF", 1.5)),
}


def apply_diffusivity_fix(param, factor):
    """Reduce Positive particle diffusivity by `factor`. Chen2020 defines
    it as a plain constant; OKane2022 defines it as a callable
    (sto, T) -> value -- handle both (see Phase 1's diagnose_diffusivity_factor.py
    for why: dividing a callable directly raises TypeError)."""
    orig = param["Positive particle diffusivity [m2.s-1]"]
    if callable(orig):
        def scaled_diffusivity(sto, T, _orig=orig, _factor=factor):
            return _orig(sto, T) / _factor
        param["Positive particle diffusivity [m2.s-1]"] = scaled_diffusivity
    else:
        param["Positive particle diffusivity [m2.s-1]"] = orig / factor
    return param


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


T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6  # verified safe in experiment 18 -- do not widen this margin


def solve_dense(param, discharge_experiment, initial_soc, chem, context=""):
    """Solve twice and splice, as validated in experiment 18/20."""
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

variations_per_size = 83

all_data = []

print(f"Starting continuous-discharge simulations (SOH fixed at {SOH_FIXED}, "
      f"C-rate {C_RATE_MIN}-{C_RATE_MAX}, cutoffs LFP={LOWER_VOLTAGE_CUTOFF['LFP']}V/NMC={LOWER_VOLTAGE_CUTOFF['NMC']}V, "
      f"NMC parameter sets {NMC_PARAMETER_SETS} (Mohtat2020 dropped), "
      f"diffusivity/{DIFFUSIVITY_FACTOR} applied to NMC's positive particle diffusivity)...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soc = random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX)
        soh = SOH_FIXED
        c_rate = random.uniform(C_RATE_MIN, C_RATE_MAX)

        ambient_c = random.uniform(15.0, 35.0)
        TEMP_RESISTANCE_COEFF = 0.02
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)

        variation_id = size_idx * variations_per_size + i

        nmc_set_name = random.choice(NMC_PARAMETER_SETS)

        print(f"\n--- Target size: {target_ah} Ah | Variation {i+1}/{variations_per_size} | "
              f"C-rate: {c_rate:.3f} | SOC: {soc:.2f} | SOH: {soh:.2f} | Ambient: {ambient_c:.1f}C | NMC set: {nmc_set_name} ---")

        param_lfp = param_lfp_base.copy()
        param_nmc = param_nmc_bases[nmc_set_name].copy()
        param_nmc = apply_diffusivity_fix(param_nmc, DIFFUSIVITY_FACTOR)

        chemistries = {
            "LFP": param_lfp,
            "NMC": param_nmc,
        }

        resistance_factors = {}
        for chem, param in chemistries.items():
            mult = capacity_multipliers[chem][size_idx]
            param["Negative electrode thickness [m]"] *= mult
            param["Positive electrode thickness [m]"] *= mult

            # SOH-scaling fix (experiment 20): scale BOTH Maximum AND
            # Initial concentration by the same factor.
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
            dense = solve_dense(param, discharge_experiment, soc, chem, context=context)
            if dense is None:
                print(f"  Skipping {chem}: no solution data.")
                continue
            time_arr, voltage_arr, capacity_arr = dense

            raw_voltage = voltage_arr
            noise = np.random.normal(0, 0.001, len(raw_voltage))
            voltage_with_noise = raw_voltage + noise

            df = pd.DataFrame({
                "Time [s]": time_arr,
                "Voltage [V]": voltage_with_noise,
                "Capacity [A.h]": capacity_arr,
                "Chemistry": chem,
                "Target_Capacity_Ah": target_ah,
                "C_Rate": round(c_rate, 4),
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
    print(f"\nDone! Diffusivity-tuned data saved to '{OUTPUT_DATA_CSV}'")
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

    plt.title(f'Simulated Continuous Discharge Profiles (SOH={SOH_FIXED}, diffusivity/{DIFFUSIVITY_FACTOR}, Mohtat2020 dropped)')
    plt.xlabel('Time [Hours]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid(True)

    plt.savefig(DISCHARGE_PLOT_PNG, dpi=150)
    print(f"Plot saved to '{DISCHARGE_PLOT_PNG}'")
    plt.close()
else:
    print("Plot skipped: no data generated.")
