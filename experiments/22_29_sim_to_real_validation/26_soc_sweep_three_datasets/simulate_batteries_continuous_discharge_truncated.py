"""
Experiment 26: the "continuous-discharge / stressed" sibling to
experiments/22_29_sim_to_real_validation/24_soh_range_0.8_1.0_sim_to_real/simulate_batteries_soh_range.py's
"rested-initialization" method. Read that script's docstring first for the
full investigation this fix came out of; this one only covers what's
different and why.

THE PROBLEM WITH THE OTHER SCRIPT, FOR THIS USE CASE: PyBaMM's
`sim.solve(initial_soc=X)` initializes a RESTED battery -- uniform
concentration profiles consistent with having sat idle at X% SOC. A real
cell that has been ACTIVELY DISCHARGING and has simply arrived at X% SOC
carries concentration gradients/polarization from that sustained current
draw that a freshly-initialized "resting at X%" cell does not have. Every
real dataset this project validates sim-to-real against (EMPA, experiment
07's set, CALCE) is measured from continuously-cycling lab cells -- never
rested-then-resumed mid-discharge -- so training on rested-low-SOC
synthetic batteries represents a scenario none of that real data actually
matches. Proven directly this session: filtering a rested-low-SOC-widened
training set back down to Initial_SOC>=0.5 (no resimulation, same battery
pool) restored EMPA balanced accuracy from ~50-65% back to ~99%, matching
the un-widened baseline exactly -- conclusive evidence the rested
initialization itself, not sampling density or numerical noise (both
independently checked and fixed first), was the cause. See
experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/RESULTS.md for the full ablation.

THIS SCRIPT'S FIX: don't use `initial_soc` to jump directly to a low SOC
at all. Instead, solve ONE full discharge per battery config starting
from (near-)full charge -- exactly like experiment 24's script already
validates for Initial_SOC~1.0 -- then apply the SAME `truncate_at_soc()`
function already used to build every real SOC-sweep dataset in this
experiment (soc_truncation.py) to slice that single full trace into all
12 target Initial_SOC variants. This makes a synthetic "battery arriving
at 10% SOC" mean the same physical thing a real one does: an actively-
discharging cell's tail end, not a rested restart. Bonus: only ONE PyBaMM
solve per battery config is needed (not one per SOC point), since all 12
variants come from slicing the same dense trace -- cheaper than any of
the three rested-initialization resimulation attempts that preceded this.

IMPORTANT -- READ BEFORE REACHING FOR EITHER SCRIPT: this is NOT the
"correct" method in general, and does not replace or obsolete
simulate_batteries_soh_range.py. Which script to use depends entirely on
what the target real-world battery has actually been doing:
  - Use THIS script (continuous-discharge-truncated) to validate against
    continuously-cycled lab datasets, like EMPA/exp07/CALCE here.
  - Use simulate_batteries_soh_range.py (rested `initial_soc`) for the
    actual Stena Recycling production model: end-of-life batteries that
    have sat idle for weeks/months before being tested cold from a rested
    low-SOC state. That IS the physically correct model for that use case
    -- do not "fix" that script to match this one.

Otherwise identical to experiment 24's script: SOH sampled from a range
per battery, C-rate range, both chemistry fixes (NMC diffusivity/10,
LFP OCP rate=-3), voltage cutoffs, capacity targets, resistance/
temperature randomization -- all copied unchanged.

Usage: env vars DATA_DIR, RUN_LABEL, OUTPUT_DATA_CSV, FAILURE_LOG_CSV,
DISCHARGE_PLOT_PNG, LFP_LOWER_CUTOFF, NMC_LOWER_CUTOFF, SOH_MIN, SOH_MAX,
C_RATE_MIN/MAX, INITIAL_SOC_FULL, DIFFUSIVITY_FACTOR, LFP_OCP_RATE_CONSTANT,
VARIATIONS_PER_SIZE (all optional).
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
sys.path.insert(0, SCRIPT_DIR)
from soc_truncation import truncate_at_soc

random.seed(42)
np.random.seed(42)

# Fixed near-1.0, not randomized: every battery needs the WIDEST possible
# capacity range available so truncate_at_soc() can slice out any of the
# 12 target SOC points, including the deepest (0.05). 0.99 rather than a
# literal 1.0 to avoid a possible solver edge case exactly at full charge.
INITIAL_SOC_FULL = float(os.environ.get("INITIAL_SOC_FULL", 0.99))
SOH_MIN = float(os.environ.get("SOH_MIN", 0.8))
SOH_MAX = float(os.environ.get("SOH_MAX", 1.0))
C_RATE_MIN = float(os.environ.get("C_RATE_MIN", 0.1))
C_RATE_MAX = float(os.environ.get("C_RATE_MAX", 0.2))
DIFFUSIVITY_FACTOR = float(os.environ.get("DIFFUSIVITY_FACTOR", 10))
LFP_OCP_RATE_CONSTANT = float(os.environ.get("LFP_OCP_RATE_CONSTANT", -3))

# Same 12-point grid this experiment's real-data build scripts use.
SOC_TRUNCATION_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "26_soh_range_continuous_discharge_truncated_v1")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
DISCHARGE_PLOT_PNG = os.environ.get("DISCHARGE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "continuous_discharge_plot.png"))

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

model = pybamm.lithium_ion.SPM()

param_lfp_base = pybamm.ParameterValues("Prada2013")
# Mohtat2020 dropped: experiment 20 Part C found diffusivity tuning
# doesn't fix its magnitude mismatch at all, unlike Chen2020/OKane2022.
NMC_PARAMETER_SETS = ["Chen2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]

LOWER_VOLTAGE_CUTOFF = {
    "LFP": float(os.environ.get("LFP_LOWER_CUTOFF", 1.5)),
    "NMC": float(os.environ.get("NMC_LOWER_CUTOFF", 1.5)),
}


def apply_nmc_diffusivity_fix(param, factor):
    """Reduce Positive particle diffusivity by `factor`. Chen2020 defines
    it as a plain constant; OKane2022 defines it as a callable
    (sto, T) -> value -- handle both."""
    orig = param["Positive particle diffusivity [m2.s-1]"]
    if callable(orig):
        def scaled_diffusivity(sto, T, _orig=orig, _factor=factor):
            return _orig(sto, T) / _factor
        param["Positive particle diffusivity [m2.s-1]"] = scaled_diffusivity
    else:
        param["Positive particle diffusivity [m2.s-1]"] = orig / factor
    return param


def make_lfp_ocp(rate_constant):
    """Same functional form as Afshar2017's LFP_ocp_Afshar2017, with the
    tail rate constant (-30 originally) replaced by `rate_constant`. An
    explicit empirical calibration -- see experiment 21's docstring."""
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


T_INTERP_N_POINTS = 3000  # fixed -- every battery here IS a full-range discharge, so
                           # the "tiny dQ" numerical issue that forced scaling this in
                           # the rested-initialization script doesn't apply here.
T_INTERP_SAFETY_FRACTION = 1 - 1e-6  # verified safe in experiments 18/20/21 -- do not widen this margin


def solve_dense(param, discharge_experiment, initial_soc, chem, context=""):
    """Solve twice and splice, as validated in experiments 18/20/21."""
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

variations_per_size = int(os.environ.get("VARIATIONS_PER_SIZE", 83))

all_data = []
full_traces_for_plot = []

print(f"Starting FULL continuous-discharge simulations from Initial_SOC={INITIAL_SOC_FULL} "
      f"(SOH sampled {SOH_MIN}-{SOH_MAX} per battery, C-rate {C_RATE_MIN}-{C_RATE_MAX}, "
      f"cutoffs LFP={LOWER_VOLTAGE_CUTOFF['LFP']}V/NMC={LOWER_VOLTAGE_CUTOFF['NMC']}V, "
      f"NMC parameter sets {NMC_PARAMETER_SETS} (Mohtat2020 dropped) with diffusivity/{DIFFUSIVITY_FACTOR}, "
      f"LFP OCP tail rate constant softened to {LFP_OCP_RATE_CONSTANT}), "
      f"then truncating each to {SOC_TRUNCATION_POINTS}...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        soh = random.uniform(SOH_MIN, SOH_MAX)
        c_rate = random.uniform(C_RATE_MIN, C_RATE_MAX)

        ambient_c = random.uniform(15.0, 35.0)
        TEMP_RESISTANCE_COEFF = 0.02
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)

        variation_id = size_idx * variations_per_size + i

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

            # SOH-scaling fix (experiment 20 Part A): scale BOTH Maximum
            # AND Initial concentration by the same factor.
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

# Combine and save
if all_data:
    training_data = pd.concat(all_data, ignore_index=True)
    training_data.to_csv(OUTPUT_DATA_CSV, index=False)
    print(f"\nDone! Continuous-discharge-truncated data saved to '{OUTPUT_DATA_CSV}'")
    print(f"  {training_data['Variation_ID'].nunique()} battery/SOC-point rows "
          f"from {training_data.groupby('Chemistry')['Variation_ID'].apply(lambda s: s.str.split('_soc_').str[0].nunique()).to_dict()} base full-discharge traces per chemistry")
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

# --- PLOTTING (full pre-truncation traces, not every truncated variant -- would be highly redundant) ---
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
        plt.plot(run_df['Time [s]'] / 3600, run_df['Voltage [V]'], color=color, alpha=0.15, linewidth=0.8, label=label)

    plt.title(f'Full Discharge Profiles Pre-Truncation (SOH {SOH_MIN}-{SOH_MAX}, NMC diffusivity/{DIFFUSIVITY_FACTOR}, '
              f'LFP OCP rate={LFP_OCP_RATE_CONSTANT}, Initial_SOC={INITIAL_SOC_FULL})')
    plt.xlabel('Time [Hours]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid(True)

    plt.savefig(DISCHARGE_PLOT_PNG, dpi=150)
    print(f"Plot saved to '{DISCHARGE_PLOT_PNG}'")
    plt.close()
else:
    print("Plot skipped: no data generated.")
