"""
optimize_empa_2 merge draft: faster synthetic-data generator. Same
physics, same job parameters, same random-draw order (seed=42) as
optimize_empa_2/simulate_batteries.py (SOH-range sampling, NMC
positive-particle diffusivity/10 fix, LFP OCP tail rate softened to
-3) -- only HOW each discharge gets solved changed, ported from
laswer5's optimize-runtime branch (which this repo's Python 3.9
environment can't run directly -- it pins Python >=3.12 and depends on
polars via uv). The two portable ideas from it, adopted here in plain
pandas/multiprocessing instead:

1. pybamm.IDAKLUSolver instead of the default solver, with ONE
   precomputed t_interp pass instead of the original solve_dense()'s
   two-pass splice (an exploratory default-density solve just to learn
   the actual discharge duration, then a second dense pass bounded by
   that). Measured directly in this environment before adopting:
   IDAKLUSolver + a single ESTIMATED t_interp array (not even requiring
   the exact duration up front) is ~2x faster per solve than the
   original two-pass approach, and verified correct -- the solver
   still terminates exactly on the voltage-cutoff event even when the
   estimated horizon undershoots the real duration (PyBaMM extends past
   the requested t_interp range as needed; confirmed empirically: a
   test solve's final time exceeded its horizon estimate by ~13% and
   still terminated exactly at the target cutoff voltage, monotonic,
   correct point count).
2. Running the (now-independent, SOC/SOH/etc-fully-specified) per-job
   solves across a worker pool (ProcessPoolExecutor) instead of serially.

Job generation (the random draws determining each battery's SOC/SOH/
C-rate/ambient/NMC-parameter-set/resistance-noise) stays serial and
seeded, in the SAME order as the original script's nested loop, so the
parameter DISTRIBUTION this produces is statistically identical to the
original -- not a byte-for-byte reproduction of its specific draws
(parallel solving makes that both impractical and pointless to chase),
but not a methodologically different sample either.

Usage: env vars VARIATIONS_PER_SIZE, SOH_MIN/MAX, C_RATE_MIN/MAX,
SOC_RANGE_MIN/MAX, DIFFUSIVITY_FACTOR, LFP_OCP_RATE_CONSTANT,
LFP_LOWER_CUTOFF, NMC_LOWER_CUTOFF, SIM_WORKERS, DATA_DIR,
OUTPUT_DATA_CSV (all optional, same defaults as simulate_batteries.py).
python3 optimize_empa_2/simulate_batteries_merged.py
"""
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

SOC_RANGE_MIN = float(os.environ.get("SOC_RANGE_MIN", 0.5))
SOC_RANGE_MAX = float(os.environ.get("SOC_RANGE_MAX", 1.0))
SOH_MIN = float(os.environ.get("SOH_MIN", 0.8))
SOH_MAX = float(os.environ.get("SOH_MAX", 1.0))
C_RATE_MIN = float(os.environ.get("C_RATE_MIN", 0.1))
C_RATE_MAX = float(os.environ.get("C_RATE_MAX", 0.2))
DIFFUSIVITY_FACTOR = float(os.environ.get("DIFFUSIVITY_FACTOR", 10))
LFP_OCP_RATE_CONSTANT = float(os.environ.get("LFP_OCP_RATE_CONSTANT", -3))
NMC_PARAMETER_SETS = ["Chen2020", "OKane2022"]
LOWER_VOLTAGE_CUTOFF = {
    "LFP": float(os.environ.get("LFP_LOWER_CUTOFF", 1.5)),
    "NMC": float(os.environ.get("NMC_LOWER_CUTOFF", 1.5)),
}
CAPACITY_TARGETS_AH = [1.2, 2.0, 3.5]
VARIATIONS_PER_SIZE = int(os.environ.get("VARIATIONS_PER_SIZE", 83))
SIM_WORKERS = int(os.environ.get("SIM_WORKERS", min(4, os.cpu_count() or 1)))
T_INTERP_N_POINTS_FULL = 3000
T_INTERP_MIN_POINTS = 100

DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_DIR = os.path.join(DATA_DIR)
OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
DISCHARGE_PLOT_PNG = os.environ.get("DISCHARGE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "continuous_discharge_plot_merged.png"))


def capacity_multiplier(base_param_name, target_ah):
    base_capacity_ah = pybamm.ParameterValues(base_param_name)["Nominal cell capacity [A.h]"]
    return target_ah / base_capacity_ah


def make_jobs():
    """Serial, seeded job generation -- same draw order per variation as
    simulate_batteries.py's loop (soc, soh, c_rate, ambient_c,
    nmc_set_name, then resistance_noise for LFP then NMC)."""
    random.seed(42)
    jobs = []
    for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
        for i in range(VARIATIONS_PER_SIZE):
            soc = random.uniform(SOC_RANGE_MIN, SOC_RANGE_MAX)
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
                    "soc": soc, "soh": soh, "c_rate": c_rate, "ambient_c": ambient_c,
                    "nmc_set_name": nmc_set_name, "resistance_factor": resistance_factor,
                    "variation_id": variation_id, "v_min": LOWER_VOLTAGE_CUTOFF[chem],
                })
    # Deterministic per-job noise seeds, independent of solve order/worker
    # assignment (parallel solving makes relying on shared global random
    # state across processes meaningless).
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
        param["Negative electrode conductivity [S.m-1]"] *= job["resistance_factor"]
        param["Positive electrode conductivity [S.m-1]"] *= job["resistance_factor"]

        current_a = job["c_rate"] * job["target_ah"]
        experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {job['v_min']} V"])

        n_points = max(T_INTERP_MIN_POINTS, round(T_INTERP_N_POINTS_FULL * job["soc"]))
        horizon_guess = max(1.0, 3600 * job["soc"] / job["c_rate"])
        times = np.linspace(0, horizon_guess, n_points)

        solver = pybamm.IDAKLUSolver(options={"num_threads": 1})
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), parameter_values=param, experiment=experiment, solver=solver)
        sol = sim.solve(initial_soc=job["soc"], t_interp=times)

        t = sol["Time [s]"].entries
        v = sol["Terminal voltage [V]"].entries
        q = sol["Discharge capacity [A.h]"].entries
        if len(t) < 5 or not np.isfinite(np.column_stack([t, v, q])).all():
            return {"status": "failed", "chem": chem, "variation_id": job["variation_id"],
                    "error": "non-finite or too-short trace"}

        noise = np.random.default_rng(job["noise_seed"]).normal(0, 0.001, len(v))
        df = pd.DataFrame({
            "Time [s]": t, "Voltage [V]": v + noise, "Capacity [A.h]": q,
            "Chemistry": chem, "Target_Capacity_Ah": job["target_ah"],
            "C_Rate": round(job["c_rate"], 4), "Size_Multiplier": mult,
            "SOH": round(job["soh"], 3), "Initial_SOC": round(job["soc"], 3),
            "Ambient_Temperature_C": round(job["ambient_c"], 2),
            "Resistance_Factor": round(job["resistance_factor"], 4),
            "Base_Parameter_Set": base_name, "V_min [V]": job["v_min"],
            "Variation_ID": job["variation_id"],
        })
        return {"status": "complete", "chem": chem, "variation_id": job["variation_id"],
                "termination": str(sol.termination), "frame": df}
    except Exception as err:
        return {"status": "failed", "chem": chem, "variation_id": job["variation_id"],
                "error": f"{type(err).__name__}: {err}"}


def main():
    for path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, DISCHARGE_PLOT_PNG):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    jobs = make_jobs()
    print(f"Generated {len(jobs)} jobs ({VARIATIONS_PER_SIZE} variations/size x "
          f"{len(CAPACITY_TARGETS_AH)} sizes x 2 chemistries). Solving with "
          f"IDAKLUSolver across {SIM_WORKERS} worker(s)...")

    t0 = time.time()
    results = []
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
    print(f"\nDone! SOH-range LFP-OCP-tuned + NMC-diffusivity-tuned data "
          f"(IDAKLU single-pass, parallelized) saved to '{OUTPUT_DATA_CSV}'")

    print("Generating continuous discharge plot...")
    plt.figure(figsize=(12, 6))
    lfp_labeled, nmc_labeled = False, False
    for (chem, var_id), run_df in training_data.groupby(["Chemistry", "Variation_ID"]):
        color = "#1f77b4" if chem == "LFP" else "#ff7f0e"
        label = None
        if chem == "LFP" and not lfp_labeled:
            label, lfp_labeled = "LFP", True
        elif chem == "NMC" and not nmc_labeled:
            label, nmc_labeled = "NMC", True
        plt.plot(run_df["Time [s]"] / 3600, run_df["Voltage [V]"], color=color, alpha=0.15, linewidth=0.8, label=label)
    plt.title(f"Simulated Discharge Profiles (SOH {SOH_MIN}-{SOH_MAX}, NMC diffusivity/{DIFFUSIVITY_FACTOR}, "
              f"LFP OCP rate={LFP_OCP_RATE_CONSTANT}) -- IDAKLU single-pass, parallelized")
    plt.xlabel("Time [Hours]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)
    plt.savefig(DISCHARGE_PLOT_PNG, dpi=150)
    print(f"Plot saved to '{DISCHARGE_PLOT_PNG}'")
    plt.close()


if __name__ == "__main__":
    main()
