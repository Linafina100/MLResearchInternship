"""
Experiment 21: does the same diffusivity-tuning fix that worked for NMC
(experiment 20, Part C) also fix LFP's own systematic dV/dQ magnitude
mismatch (documented since experiment 07, never previously
investigated)?

A single-point plan-mode-style check (Prada2013, SOH=0.8, 0.15C,
resistance_factor=0.8) found, surprisingly: `Positive particle
diffusivity` (LFP's own active-material parameter) has ZERO measurable
effect on target-zone dV/dQ across a 500x range (factor 1/50 to x10 --
mean stays at -3.38 throughout, vs. real LFP's -0.3 to -2.9 target).
Exchange-current density (positive electrode) also has zero effect
across a 100x range. Negative electrode (graphite) diffusivity moves
the magnitude the WRONG direction (more negative, away from target) as
it's reduced. This is a genuinely different picture from NMC, where
diffusivity was the dominant, clearly-working lever -- consistent with
LFP's well-known two-phase (not single-phase diffusion-limited)
lithiation mechanism: the flat-plateau-then-tail shape is a
thermodynamic property of the fitted OCP curve, not primarily governed
by transport/kinetic parameters the way NMC's smooth single-phase curve
is.

This script confirms the diffusivity null result systematically across
the real C-rate (0.1/0.15/0.2) and temperature (15/25/35C) ranges at
SOH=0.8 (matching experiment 20's diagnostic structure), rather than
trusting the single-point check alone.

Usage: python3 experiments/21_lfp_diffusivity_tuning/diagnose_lfp_diffusivity_factor.py
"""
import pybamm
import numpy as np
import pandas as pd

V_MIN = 1.5
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
REAL_LFP_TARGET_MIN, REAL_LFP_TARGET_MAX = -2.9, -0.3  # experiment 20 Part A's own real-data measurement
OUTLIER_DVDQ_THRESHOLD = 50.0
T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6  # verified safe in experiment 18/20 -- do not widen this margin

DIFFUSIVITY_FACTORS = [1, 5, 10, 20, 50]  # 1 = baseline, no reduction
C_RATE_VALUES = [0.1, 0.15, 0.2]
AMBIENT_TEMP_VALUES = [15.0, 25.0, 35.0]
TARGET_AH = 2.0
SOH = 0.8
INITIAL_SOC = 1.0
TEMP_RESISTANCE_COEFF = 0.02
RESISTANCE_NOISE = 1.0

model = pybamm.lithium_ion.SPM()
param_lfp_base = pybamm.ParameterValues("Prada2013")


def build_param(diffusivity_factor, ambient_c):
    param = param_lfp_base.copy()
    base_capacity_ah = param_lfp_base["Nominal cell capacity [A.h]"]
    mult = TARGET_AH / base_capacity_ah
    param["Negative electrode thickness [m]"] *= mult
    param["Positive electrode thickness [m]"] *= mult

    param["Maximum concentration in negative electrode [mol.m-3]"] *= SOH
    param["Maximum concentration in positive electrode [mol.m-3]"] *= SOH
    param["Initial concentration in negative electrode [mol.m-3]"] *= SOH
    param["Initial concentration in positive electrode [mol.m-3]"] *= SOH

    temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)
    resistance_factor = min(1.0, max(0.3, SOH * RESISTANCE_NOISE * temp_conductivity_multiplier))
    param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
    param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

    orig = param["Positive particle diffusivity [m2.s-1]"]
    if callable(orig):
        def scaled_diffusivity(sto, T, _orig=orig, _factor=diffusivity_factor):
            return _orig(sto, T) / _factor
        param["Positive particle diffusivity [m2.s-1]"] = scaled_diffusivity
    else:
        param["Positive particle diffusivity [m2.s-1]"] = orig / diffusivity_factor
    return param


def transitions(time, voltage, capacity):
    order = np.argsort(time, kind="stable")
    t, v, q = time[order], voltage[order], capacity[order]
    dV, dQ = np.diff(v), np.diff(q)
    valid = dQ > 1e-5
    return v[1:][valid], (dV[valid] / dQ[valid])


def simulate_one(diffusivity_factor, c_rate, ambient_c):
    try:
        param = build_param(diffusivity_factor, ambient_c)
    except Exception as err:
        return {"diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED build_param ({type(err).__name__}): {err}"}

    current_a = c_rate * TARGET_AH
    experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {V_MIN} V"])

    sim1 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    try:
        sol1 = sim1.solve(initial_soc=INITIAL_SOC)
    except Exception as err:
        return {"diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED default solve ({type(err).__name__}): {err}"}

    time_default = sol1["Time [s]"].entries
    voltage_default = sol1["Terminal voltage [V]"].entries
    capacity_default = sol1["Discharge capacity [A.h]"].entries
    if len(time_default) < 5:
        return {"diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": "FAILED (too few default points)"}

    tf = time_default[-1]
    t_bound = tf * T_INTERP_SAFETY_FRACTION

    sim2 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    t_interp = np.linspace(0, t_bound, T_INTERP_N_POINTS)
    try:
        sol2 = sim2.solve(initial_soc=INITIAL_SOC, t_interp=t_interp)
    except Exception as err:
        return {"diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED t_interp solve ({type(err).__name__}): {err}"}

    time_dense = sol2["Time [s]"].entries
    voltage_dense = sol2["Terminal voltage [V]"].entries
    capacity_dense = sol2["Discharge capacity [A.h]"].entries

    after_mask = time_default > time_dense[-1]
    time_spliced = np.concatenate([time_dense, time_default[after_mask]])
    voltage_spliced = np.concatenate([voltage_dense, voltage_default[after_mask]])
    capacity_spliced = np.concatenate([capacity_dense, capacity_default[after_mask]])

    v_curr, dvdq = transitions(time_spliced, voltage_spliced, capacity_spliced)

    in_zone = (v_curr >= TARGET_ZONE_MIN) & (v_curr <= TARGET_ZONE_MAX)
    n_zone = int(in_zone.sum())
    if n_zone == 0:
        return {"diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": "COLLAPSED (0 target-zone points)"}

    zone_vals = dvdq[in_zone]
    n_outliers = int((np.abs(zone_vals) > OUTLIER_DVDQ_THRESHOLD).sum())

    return {
        "diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
        "status": "OK", "n_zone": n_zone, "mean_dvdq": float(zone_vals.mean()),
        "min_dvdq": float(zone_vals.min()), "max_dvdq": float(zone_vals.max()),
        "n_outliers_zone": n_outliers,
    }


def main():
    results = []
    for factor in DIFFUSIVITY_FACTORS:
        for c_rate in C_RATE_VALUES:
            for ambient_c in AMBIENT_TEMP_VALUES:
                r = simulate_one(factor, c_rate, ambient_c)
                results.append(r)
                tag = "OK" if r["status"] == "OK" else r["status"]
                print(f"factor={factor:<3} C-rate={c_rate:<4} T={ambient_c:<5} -> {tag}"
                      + (f"  mean_dvdq={r.get('mean_dvdq'):.2f}  n_zone={r.get('n_zone')}  "
                         f"outliers={r.get('n_outliers_zone')}"
                         if r["status"] == "OK" else ""))

    df = pd.DataFrame(results)
    out_csv = "experiments/21_lfp_diffusivity_tuning/diagnostic_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to '{out_csv}'")

    print(f"\n{'=' * 80}\nSUMMARY: per diffusivity factor -- robustness and magnitude (real target: -2.9 to -0.3)\n{'=' * 80}")
    for factor in DIFFUSIVITY_FACTORS:
        subset = df[df["diffusivity_factor"] == factor]
        n_total = len(subset)
        n_failed = (subset["status"] != "OK").sum()
        ok = subset[subset["status"] == "OK"]
        n_target_hit = 0
        mean_mag = None
        if len(ok):
            n_target_hit = ((ok["mean_dvdq"] >= REAL_LFP_TARGET_MIN) & (ok["mean_dvdq"] <= REAL_LFP_TARGET_MAX)).sum()
            mean_mag = ok["mean_dvdq"].mean()
        print(f"  factor={factor:<3}: {n_total - n_failed}/{n_total} robust | "
              f"{n_target_hit}/{len(ok)} land in real target (-2.9 to -0.3) | "
              f"mean magnitude={mean_mag if mean_mag is None else round(mean_mag, 2)}")


if __name__ == "__main__":
    main()
