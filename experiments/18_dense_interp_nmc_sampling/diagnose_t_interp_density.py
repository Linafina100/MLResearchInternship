"""
Experiment 18 diagnostic: does PyBaMM's `t_interp` (intra-solve dense
interpolation, supported by the default IDAKLUSolver) actually increase
synthetic NMC's raw-sample density in the 2.5-3.0V real-world target
zone -- the problem experiments 16/17 found is untouched by cutoff, SOH,
or C-rate?

Method, per battery:
1. Solve the experiment normally (default adaptive output) -- this is
   the baseline, and also supplies the genuine final segment (the
   solver-termination-adjacent points that exclude_final_transition /
   the termination-artifact handling needs untouched).
2. Re-solve the SAME experiment with `t_interp` set to a dense, uniform
   grid from 0 up to 95% of the first solve's actual discharge duration
   (t_interp crashes if its upper bound reaches/exceeds the real
   event-triggered termination time, which isn't known in advance --
   verified in this session; the 95% margin avoids that while only
   sacrificing the last few genuine points, which step 3 restores).
3. Splice: dense points (from step 2) covering [0, 0.95*tf], followed by
   the ORIGINAL solve's own points that fall after that cutoff (i.e. the
   genuine terminal segment, unmodified) -- so density increases only in
   the well-behaved mid-curve region, and the actual event-adjacent
   behavior (and whatever the existing artifact-handling logic expects
   to find there) is left exactly as before.
4. Compute dV/dQ transitions on both the default-only trace and the
   spliced trace (same method as feature_engineering.py: consecutive-
   sample pairs, dQ > 1e-5, binned by ending voltage), and compare
   how many land in the 2.5-3.0V zone, plus check the "seam" transition
   (last dense point -> first restored real point) isn't an artificial
   outlier.

Usage: python3 experiments/18_dense_interp_nmc_sampling/diagnose_t_interp_density.py
"""
import pybamm
import numpy as np
import pandas as pd

V_MIN = 1.5
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
OUTLIER_DVDQ_THRESHOLD = 50.0
T_INTERP_N_POINTS = 3000
# Verified in this session: t_interp crashes only if its bound reaches/
# exceeds the true event time (a naive 0.95*tf margin was tried first and
# found WAY too conservative -- it silently excluded genuine target-zone
# points for several combos, since PyBaMM auto-appends the true final
# point regardless of the requested bound, so there is no need to leave a
# large safety margin at all). This much tighter bound was confirmed safe
# (no crash) across all combos tested.
T_INTERP_SAFETY_FRACTION = 1 - 1e-6

NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
SOH_VALUES = [0.6, 0.8, 1.0]
C_RATE_VALUES = [0.2, 0.6, 1.0]
TARGET_AH = 2.0
INITIAL_SOC = 1.0

model = pybamm.lithium_ion.SPM()
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}


def build_param(base_params, soh, target_ah):
    param = base_params.copy()
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    mult = target_ah / base_capacity_ah
    param["Negative electrode thickness [m]"] *= mult
    param["Positive electrode thickness [m]"] *= mult
    param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
    param["Maximum concentration in positive electrode [mol.m-3]"] *= soh
    return param


def transitions(time, voltage, capacity):
    """Per-transition (ending voltage, dV/dQ) for consecutive samples with dQ > 1e-5,
    matching feature_engineering.py's method exactly."""
    order = np.argsort(time, kind="stable")
    t, v, q = time[order], voltage[order], capacity[order]
    dV, dQ = np.diff(v), np.diff(q)
    valid = dQ > 1e-5
    return v[1:][valid], (dV[valid] / dQ[valid])


def zone_stats(v_curr, dvdq):
    in_zone = (v_curr >= TARGET_ZONE_MIN) & (v_curr < TARGET_ZONE_MAX)
    n_in_zone = int(in_zone.sum())
    zone_vals = dvdq[in_zone]
    max_abs = float(np.max(np.abs(zone_vals))) if n_in_zone else None
    n_outliers = int((np.abs(zone_vals) > OUTLIER_DVDQ_THRESHOLD).sum()) if n_in_zone else 0
    return n_in_zone, max_abs, n_outliers


def simulate_one(label, base_params, soh, c_rate):
    param = build_param(base_params, soh, TARGET_AH)
    current_a = c_rate * TARGET_AH
    experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {V_MIN} V"])

    # Step 1: default solve
    sim1 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    try:
        sol1 = sim1.solve(initial_soc=INITIAL_SOC)
    except pybamm.SolverError as err:
        sol1 = getattr(sim1, "solution", None)
        if sol1 is None or len(sol1.t) == 0:
            return {"label": label, "soh": soh, "c_rate": c_rate, "status": f"FAILED (SolverError, no data): {err}"}
    except Exception as err:
        return {"label": label, "soh": soh, "c_rate": c_rate, "status": f"FAILED ({type(err).__name__}): {err}"}

    time_default = sol1["Time [s]"].entries
    voltage_default = sol1["Terminal voltage [V]"].entries
    capacity_default = sol1["Discharge capacity [A.h]"].entries
    if len(time_default) < 5:
        return {"label": label, "soh": soh, "c_rate": c_rate, "status": "FAILED (too few default points)"}

    tf = time_default[-1]
    t_bound = tf * T_INTERP_SAFETY_FRACTION

    # Step 2: dense t_interp solve, bounded safely below the real event time
    sim2 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    t_interp = np.linspace(0, t_bound, T_INTERP_N_POINTS)
    try:
        sol2 = sim2.solve(initial_soc=INITIAL_SOC, t_interp=t_interp)
    except Exception as err:
        return {"label": label, "soh": soh, "c_rate": c_rate, "status": f"FAILED (t_interp solve, {type(err).__name__}): {err}"}

    time_dense = sol2["Time [s]"].entries
    voltage_dense = sol2["Terminal voltage [V]"].entries
    capacity_dense = sol2["Discharge capacity [A.h]"].entries

    # Step 3: splice -- dense points up to t_bound, then the ORIGINAL
    # solve's own points after that (the genuine terminal segment).
    after_mask = time_default > time_dense[-1]
    time_spliced = np.concatenate([time_dense, time_default[after_mask]])
    voltage_spliced = np.concatenate([voltage_dense, voltage_default[after_mask]])
    capacity_spliced = np.concatenate([capacity_dense, capacity_default[after_mask]])

    # Step 4: compare
    v_curr_default, dvdq_default = transitions(time_default, voltage_default, capacity_default)
    v_curr_spliced, dvdq_spliced = transitions(time_spliced, voltage_spliced, capacity_spliced)

    n_zone_default, max_dvdq_default, n_outliers_default = zone_stats(v_curr_default, dvdq_default)
    n_zone_spliced, max_dvdq_spliced, n_outliers_spliced = zone_stats(v_curr_spliced, dvdq_spliced)

    # Seam transition: last dense point -> first restored real point (if any real points were appended)
    seam_dvdq = None
    if after_mask.sum() > 0:
        dV_seam = voltage_default[after_mask][0] - voltage_dense[-1]
        dQ_seam = capacity_default[after_mask][0] - capacity_dense[-1]
        seam_dvdq = float(dV_seam / dQ_seam) if dQ_seam > 1e-9 else None

    return {
        "label": label, "soh": soh, "c_rate": c_rate, "status": "OK",
        "tf": round(float(tf), 1),
        "n_points_default": len(time_default), "n_points_spliced": len(time_spliced),
        "n_zone_default": n_zone_default, "n_zone_spliced": n_zone_spliced,
        "max_dvdq_zone_default": max_dvdq_default, "max_dvdq_zone_spliced": max_dvdq_spliced,
        "n_outliers_zone_default": n_outliers_default, "n_outliers_zone_spliced": n_outliers_spliced,
        "seam_dvdq": seam_dvdq,
    }


def main():
    results = []
    for nmc_set in NMC_PARAMETER_SETS:
        for soh in SOH_VALUES:
            for c_rate in C_RATE_VALUES:
                print(f"\n-- {nmc_set}, SOH={soh}, C-rate={c_rate} --")
                r = simulate_one(nmc_set, param_nmc_bases[nmc_set], soh, c_rate)
                print(f"   {r}")
                results.append(r)

    df = pd.DataFrame(results)
    out_csv = "experiments/18_dense_interp_nmc_sampling/diagnostic_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to '{out_csv}'")

    ok = df[df["status"] == "OK"]
    failed = df[df["status"] != "OK"]
    print(f"\n{'=' * 70}\nSUMMARY ({len(ok)}/{len(df)} solves succeeded)\n{'=' * 70}")
    if len(failed):
        print("Failures:")
        print(failed[["label", "soh", "c_rate", "status"]].to_string(index=False))

    print(f"\nTotal target-zone (2.5-3.0V) points -- default: {ok['n_zone_default'].sum()}, "
          f"spliced: {ok['n_zone_spliced'].sum()}")
    print(f"Batteries with >=1 genuine target-zone point -- default: {(ok['n_zone_default'] > 0).sum()}/{len(ok)}, "
          f"spliced: {(ok['n_zone_spliced'] > 0).sum()}/{len(ok)}")
    print(f"Outlier transitions in target zone -- default: {ok['n_outliers_zone_default'].sum()}, "
          f"spliced: {ok['n_outliers_zone_spliced'].sum()}")
    print(f"Max |dV/dQ| in target zone -- default: {ok['max_dvdq_zone_default'].max()}, "
          f"spliced: {ok['max_dvdq_zone_spliced'].max()}")
    seam_vals = ok["seam_dvdq"].dropna()
    print(f"\nSeam transition |dV/dQ| stats (checking the splice itself isn't an artifact, "
          f"n={len(seam_vals)} non-null of {len(ok)}): "
          f"max={seam_vals.abs().max() if len(seam_vals) else 'n/a'}, "
          f"mean={seam_vals.abs().mean() if len(seam_vals) else float('nan'):.3f}")

    print("\nPer-parameter-set breakdown (spliced target-zone points):")
    print(ok.groupby("label")["n_zone_spliced"].agg(["sum", "mean"]))


if __name__ == "__main__":
    main()
