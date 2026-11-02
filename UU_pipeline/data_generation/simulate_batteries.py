"""
optimize_empa, experiment 06 (pooled model, full SOC range): abandons
per-SOC specialization entirely -- the actual production goal is ONE
model that predicts chemistry regardless of the battery's unknown
test-time SOC, not 12 separately-trained specialists. Per-SOC
specialization (attempts 01-05, inherited from a colleague's branch)
also turned out to be why Random Forest collapsed to ~50% balanced
accuracy while XGBoost didn't: it shrank each bucket's training set to
~100-500 rows, which RandomForestClassifier(max_depth=10) overfits on.
A genuinely pooled model trained on a much larger set (experiment 22's
97.89% RF result, ~5,000+ rows) doesn't have that problem.

The blocker for pooling: every prior attempt's generator only produces
data for Initial_SOC 0.5-1.0 (SOC_RANGE_MIN=0.5), via
`sim.solve(initial_soc=X)` -- a REST-initialized simulation (battery
sat idle at X% then tested cold). Experiment 26 directly proved that
mixing REST-initialized LOW-SOC data into training collapses accuracy
to ~50-65% against every real dataset this project validates against
(EMPA/exp07/CALCE), none of which are ever rested-then-resumed -- so
simply widening SOC_RANGE_MIN downward in that script would reintroduce
exactly that bug.

THIS SCRIPT'S FIX (ported unchanged from
experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/simulate_batteries_continuous_discharge_truncated.py):
solve ONE full discharge per battery config from near-full charge, then
slice that single trace at all 12 target Initial_SOC points with
`truncate_at_soc()` (soc_truncation.py, local copy) -- the same method
every real build script here already uses on real cycles. This also
means only one solve per job, not one per SOC point.

Kept unchanged from attempt 03's generator (itself based on
optimize_empa_2/simulate_batteries_merged.py): NMC positive-particle
diffusivity/10 fix, LFP OCP tail rate softened to -3, SOH-scaling on
both electrodes, the validated NEG_CAPACITY_FACTOR=0.45 LFP-only
negative-electrode rebalance, resistance/conductivity scaling,
IDAKLUSolver, ProcessPoolExecutor-parallelized job solving, seed=42
deterministic job generation.

Two additions:
1. THREAD CAP (ported from Laswer5's optimize-runtime branch -- the one
   piece of it that's portable without Polars/Python>=3.12, confirmed
   not available in this environment): cap OMP/OPENBLAS/MKL/VECLIB
   thread counts to 1 before numpy/pybamm import, so each
   ProcessPoolExecutor worker's own BLAS calls don't additionally
   spawn competing threads on top of the worker-process parallelism
   itself (IDAKLUSolver's own `num_threads: 1` option only caps the
   solver, not the underlying BLAS library).
2. SIM_WORKERS now defaults to every available core (was capped at 4)
   -- safe now that each worker is single-threaded.

Writes to this campaign's own data/optimize_empa_low_soc_06_synthetic/
namespace.

Usage: env vars VARIATIONS_PER_SIZE, SOH_MIN/MAX, C_RATE_MIN/MAX,
DIFFUSIVITY_FACTOR, LFP_OCP_RATE_CONSTANT, NEG_CAPACITY_FACTOR,
LFP_LOWER_CUTOFF, NMC_LOWER_CUTOFF, SIM_WORKERS, DATA_DIR,
OUTPUT_DATA_CSV, INITIAL_SOC_FULL (all optional).
python3 optimize_empa/experiment/06_pooled_model_full_soc_range/data_generation/simulate_batteries.py
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, SCRIPT_DIR)
from soc_truncation import truncate_at_soc

# Fixed near-1.0, not randomized -- every battery needs the WIDEST
# possible capacity range so truncate_at_soc() can slice out any of the
# 12 target points, including the deepest (0.05). 0.99 rather than a
# literal 1.0 to avoid a possible solver edge case exactly at full charge.
INITIAL_SOC_FULL = float(os.environ.get("INITIAL_SOC_FULL", 0.99))
SOC_TRUNCATION_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

SOH_MIN = float(os.environ.get("SOH_MIN", 0.8))
SOH_MAX = float(os.environ.get("SOH_MAX", 1.0))
C_RATE_MIN = float(os.environ.get("C_RATE_MIN", 0.1))
C_RATE_MAX = float(os.environ.get("C_RATE_MAX", 0.2))
DIFFUSIVITY_FACTOR = float(os.environ.get("DIFFUSIVITY_FACTOR", 10))
LFP_OCP_RATE_CONSTANT = float(os.environ.get("LFP_OCP_RATE_CONSTANT", -3))
NEG_CAPACITY_FACTOR = float(os.environ.get("NEG_CAPACITY_FACTOR", 0.45))
NMC_PARAMETER_SETS = ["Chen2020", "OKane2022"]
LOWER_VOLTAGE_CUTOFF = {
    "LFP": float(os.environ.get("LFP_LOWER_CUTOFF", 1.5)),
    "NMC": float(os.environ.get("NMC_LOWER_CUTOFF", 1.5)),
}
CAPACITY_TARGETS_AH = [1.2, 2.0, 3.5]
VARIATIONS_PER_SIZE = int(os.environ.get("VARIATIONS_PER_SIZE", 83))
SIM_WORKERS = int(os.environ.get("SIM_WORKERS", os.cpu_count() or 1))

# Fixed, not scaled by soc -- every job here IS a full-range discharge
# (unlike the rest-initialized scripts, where a low initial_soc meant
# spreading the same point count over a much smaller capacity range).
T_INTERP_N_POINTS = 3000

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_DIR, "data", "optimize_empa_low_soc_06_synthetic"))
RUN_DIR = os.path.join(DATA_DIR)
OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
DISCHARGE_PLOT_PNG = os.environ.get("DISCHARGE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "continuous_discharge_plot.png"))


def capacity_multiplier(base_param_name, target_ah):
    base_capacity_ah = pybamm.ParameterValues(base_param_name)["Nominal cell capacity [A.h]"]
    return target_ah / base_capacity_ah


def make_jobs():
    """Serial, seeded job generation -- same draw order per variation as
    attempt 03's generator (soc is no longer drawn; everything else
    unchanged), so the parameter distribution stays comparable."""
    random.seed(42)
    jobs = []
    for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
        for i in range(VARIATIONS_PER_SIZE):
            soh = random.uniform(SOH_MIN, SOH_MAX)
            c_rate = random.uniform(C_RATE_MIN, C_RATE_MAX)
            ambient_c = random.uniform(15.0, 35.0)
            nmc_set_name = random.choice(NMC_PARAMETER_SETS)
            temp_conductivity_multiplier = 1.0 + 0.02 * (ambient_c - 25.0)
            variation_id = size_idx * VARIATIONS_PER_SIZE + i

            for chem in ("LFP", "NMC"):
                resistance_noise = random.uniform(0.8, 1.2)
                resistance_factor = min(1.0, max(0.3, soh * resistance_noise * temp_conductivity_multiplier))
                jobs.append({
                    "chem": chem, "target_ah": target_ah, "size_idx": size_idx,
                    "soh": soh, "c_rate": c_rate, "ambient_c": ambient_c,
                    "nmc_set_name": nmc_set_name, "resistance_factor": resistance_factor,
                    "variation_id": variation_id, "v_min": LOWER_VOLTAGE_CUTOFF[chem],
                })
    # Deterministic per-job noise seeds, independent of solve order/worker
    # assignment.
    for job, seed in zip(jobs, np.random.SeedSequence(42).spawn(len(jobs))):
        job["noise_seed"] = int(seed.generate_state(1)[0])
    return jobs


def solve_job(job):
    chem = job["chem"]
    try:
        if chem == "LFP":
            param = pybamm.ParameterValues("Prada2013").copy()
            rate = LFP_OCP_RATE_CONSTANT
            param["Positive electrode OCP [V]"] = (
                lambda sto, _r=rate: 3.4077 - 0.020269 * sto + 0.5 * np.exp(-150 * sto) - 0.9 * np.exp(_r * (1 - sto))
            )
            base_name = "Prada2013"
        else:
            param = pybamm.ParameterValues(job["nmc_set_name"]).copy()
            orig = param["Positive particle diffusivity [m2.s-1]"]
            factor = DIFFUSIVITY_FACTOR
            if callable(orig):
                param["Positive particle diffusivity [m2.s-1]"] = (
                    lambda sto, temp, _orig=orig, _f=factor: _orig(sto, temp) / _f
                )
            else:
                param["Positive particle diffusivity [m2.s-1]"] = orig / factor
            base_name = job["nmc_set_name"]

        mult = capacity_multiplier(base_name, job["target_ah"])
        param["Negative electrode thickness [m]"] *= mult
        param["Positive electrode thickness [m]"] *= mult
        param["Maximum concentration in negative electrode [mol.m-3]"] *= job["soh"]
        param["Maximum concentration in positive electrode [mol.m-3]"] *= job["soh"]
        param["Initial concentration in negative electrode [mol.m-3]"] *= job["soh"]
        param["Initial concentration in positive electrode [mol.m-3]"] *= job["soh"]

        if chem == "LFP":
            # Validated negative-electrode capacity rebalance (attempts
            # 02/03) -- independent of the positive electrode's own
            # SOH scaling. See optimize_empa/experiment/02_negative_electrode_balance/RESULTS.md.
            param["Maximum concentration in negative electrode [mol.m-3]"] *= NEG_CAPACITY_FACTOR
            param["Initial concentration in negative electrode [mol.m-3]"] *= NEG_CAPACITY_FACTOR

        param["Negative electrode conductivity [S.m-1]"] *= job["resistance_factor"]
        param["Positive electrode conductivity [S.m-1]"] *= job["resistance_factor"]

        current_a = job["c_rate"] * job["target_ah"]
        experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {job['v_min']} V"])

        horizon_guess = max(1.0, 3600 * INITIAL_SOC_FULL / job["c_rate"])
        times = np.linspace(0, horizon_guess, T_INTERP_N_POINTS)

        solver = pybamm.IDAKLUSolver(options={"num_threads": 1})
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), parameter_values=param, experiment=experiment, solver=solver)
        sol = sim.solve(initial_soc=INITIAL_SOC_FULL, t_interp=times)

        t = sol["Time [s]"].entries
        v = sol["Terminal voltage [V]"].entries
        q = sol["Discharge capacity [A.h]"].entries
        if len(t) < 5 or not np.isfinite(np.column_stack([t, v, q])).all():
            return {"status": "failed", "chem": chem, "variation_id": job["variation_id"],
                    "error": "non-finite or too-short trace"}

        noise = np.random.default_rng(job["noise_seed"]).normal(0, 0.001, len(v))
        v_noisy = v + noise

        base_row = {
            "Chemistry": chem, "Target_Capacity_Ah": job["target_ah"],
            "C_Rate": round(job["c_rate"], 4), "Size_Multiplier": mult,
            "SOH": round(job["soh"], 3), "Ambient_Temperature_C": round(job["ambient_c"], 2),
            "Resistance_Factor": round(job["resistance_factor"], 4),
            "Base_Parameter_Set": base_name, "V_min [V]": job["v_min"],
        }

        frames = []
        for s in SOC_TRUNCATION_POINTS:
            truncated = truncate_at_soc(t, v_noisy, q, s)
            if truncated is None:
                continue
            t_trunc, v_trunc, q_trunc = truncated
            frames.append(pd.DataFrame({
                "Time [s]": t_trunc, "Voltage [V]": v_trunc, "Capacity [A.h]": q_trunc,
                "Initial_SOC": s, "Variation_ID": f"{job['variation_id']}_soc_{s}",
                **base_row,
            }))

        if not frames:
            return {"status": "failed", "chem": chem, "variation_id": job["variation_id"],
                    "error": "no SOC truncations survived"}

        return {"status": "complete", "chem": chem, "variation_id": job["variation_id"],
                "termination": str(sol.termination), "frame": pd.concat(frames, ignore_index=True),
                "full_trace": pd.DataFrame({"Time [s]": t, "Voltage [V]": v_noisy,
                                             "Chemistry": chem, "Variation_ID": job["variation_id"]})}
    except Exception as err:
        return {"status": "failed", "chem": chem, "variation_id": job["variation_id"],
                "error": f"{type(err).__name__}: {err}"}


def main():
    for path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    jobs = make_jobs()
    print(f"Generated {len(jobs)} jobs ({VARIATIONS_PER_SIZE} variations/size x "
          f"{len(CAPACITY_TARGETS_AH)} sizes x 2 chemistries). Each solves ONE full "
          f"discharge from Initial_SOC={INITIAL_SOC_FULL}, then truncates at "
          f"{SOC_TRUNCATION_POINTS}. Solving with IDAKLUSolver across {SIM_WORKERS} worker(s)...")

    t0 = time.time()
    if SIM_WORKERS <= 1:
        results = [solve_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=SIM_WORKERS) as pool:
            results = list(pool.map(solve_job, jobs))
    elapsed = time.time() - t0

    failures = [r for r in results if r["status"] != "complete"]
    completed = [r for r in results if r["status"] == "complete"]
    print(f"\n{len(completed)}/{len(jobs)} solves completed, {len(failures)} failed, in {elapsed:.1f}s "
          f"({elapsed/len(jobs):.3f}s/job average, {SIM_WORKERS} workers).")

    if failures:
        failures_df = pd.DataFrame([{"Chemistry": f["chem"], "Variation_ID": f["variation_id"],
                                      "Error": f["error"]} for f in failures])
        print(failures_df["Error"].value_counts().to_string())
        failures_df.to_csv(FAILURE_LOG_CSV, index=False)
        print(f"Full failure log saved to '{FAILURE_LOG_CSV}'")

    if not completed:
        print("\nNo data generated: every solve attempt failed.")
        return

    training_data = pd.concat([r["frame"] for r in completed], ignore_index=True)
    training_data.to_csv(OUTPUT_DATA_CSV, index=False)
    n_base_traces = training_data.groupby("Chemistry")["Variation_ID"].apply(
        lambda s: s.str.split("_soc_").str[0].nunique()).to_dict()
    print(f"\nDone! Continuous-discharge-truncated, pooled-model data (NMC diffusivity/{DIFFUSIVITY_FACTOR}, "
          f"LFP OCP rate={LFP_OCP_RATE_CONSTANT}, NEG_CAPACITY_FACTOR={NEG_CAPACITY_FACTOR}) saved to "
          f"'{OUTPUT_DATA_CSV}': {len(training_data)} rows from {n_base_traces} base full-discharge traces/chemistry.")

    print("Generating continuous discharge plot (full pre-truncation traces)...")
    plot_df = pd.concat([r["full_trace"] for r in completed], ignore_index=True)
    plt.figure(figsize=(12, 6))
    lfp_labeled, nmc_labeled = False, False
    for (chem, var_id), run_df in plot_df.groupby(["Chemistry", "Variation_ID"]):
        color = "#1f77b4" if chem == "LFP" else "#ff7f0e"
        label = None
        if chem == "LFP" and not lfp_labeled:
            label, lfp_labeled = "LFP", True
        elif chem == "NMC" and not nmc_labeled:
            label, nmc_labeled = "NMC", True
        plt.plot(run_df["Time [s]"] / 3600, run_df["Voltage [V]"], color=color, alpha=0.15, linewidth=0.8, label=label)
    plt.title(f"Full Discharge Profiles Pre-Truncation (SOH {SOH_MIN}-{SOH_MAX}, NMC diffusivity/{DIFFUSIVITY_FACTOR}, "
              f"LFP OCP rate={LFP_OCP_RATE_CONSTANT}, NEG_CAPACITY_FACTOR={NEG_CAPACITY_FACTOR})")
    plt.xlabel("Time [Hours]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)
    plt.savefig(DISCHARGE_PLOT_PNG, dpi=150)
    print(f"Plot saved to '{DISCHARGE_PLOT_PNG}'")
    plt.close()


if __name__ == "__main__":
    main()
