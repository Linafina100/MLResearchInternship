"""
Experiment 22, Phase 1 diagnostic: find a robust `Positive particle
diffusivity` reduction factor that brings synthetic NMC's target-zone
(2.5-3.0V) dV/dQ magnitude down toward real NMC's -3 to -8 range
(experiment 20's own measurement), across the actual randomized
conditions this project's pipeline uses -- not just the single point
checked in this experiment's plan-mode investigation.

That investigation found (single point: Chen2020, SOH=0.8, 0.15C,
resistance_factor=0.8): reducing electrode conductivity or contact
resistance has ZERO effect on dV/dQ (a resistive term only shifts the
curve's level under ~constant current, never its slope -- SPM doesn't
even wire electrode conductivity into the solved dynamics). Reducing the
reaction rate (exchange-current density) has a real but weak effect.
Reducing solid-state diffusivity is the one lever that works
mechanistically (slower particle-surface concentration tracking smears
the voltage transition) -- but the safe window is narrow: 10x reduction
worked well, 15x already degraded badly (a single extreme-outlier
point), 20x+ collapsed to zero target-zone points entirely (the same
failure mode experiments 18/20 characterized, now diffusivity-driven).

This script sweeps candidate factors across all 3 NMC parameter sets
(Chen2020/Mohtat2020/OKane2022 -- Mohtat2020 has behaved differently
from the other two in every prior experiment) and the real C-rate
(0.1-0.2)/temperature (15-35C) ranges at SOH=0.8, using the same dense
`t_interp`-and-splice method as experiments 18/20, checking: no solve
failures, no collapse to zero target-zone points, and how close the
resulting magnitude gets to real NMC's -3 to -8 target -- plus a check
that bins OUTSIDE the target zone aren't badly distorted.

Usage: python3 experiments/22_nmc_diffusivity_tuning/diagnose_diffusivity_factor.py
"""
import pybamm
import numpy as np
import pandas as pd

V_MIN = 1.5
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
REAL_NMC_TARGET_MIN, REAL_NMC_TARGET_MAX = -8.0, -3.0  # experiment 20's own real-data measurement
OUTLIER_DVDQ_THRESHOLD = 50.0
T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6  # verified safe in experiment 18 -- do not widen this margin

NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
DIFFUSIVITY_FACTORS = [1, 4, 6, 8, 10, 12]  # 1 = baseline, no reduction
C_RATE_VALUES = [0.1, 0.15, 0.2]
AMBIENT_TEMP_VALUES = [15.0, 25.0, 35.0]
TARGET_AH = 2.0
SOH = 0.8
INITIAL_SOC = 1.0
TEMP_RESISTANCE_COEFF = 0.02
RESISTANCE_NOISE = 1.0  # fixed, matching experiment 18's mild-conditions convention

model = pybamm.lithium_ion.SPM()
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}


def build_param(base_params, diffusivity_factor, ambient_c):
    param = base_params.copy()
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    mult = TARGET_AH / base_capacity_ah
    param["Negative electrode thickness [m]"] *= mult
    param["Positive electrode thickness [m]"] *= mult

    # SOH-scaling fix (experiment 20): scale both Maximum and Initial concentration.
    param["Maximum concentration in negative electrode [mol.m-3]"] *= SOH
    param["Maximum concentration in positive electrode [mol.m-3]"] *= SOH
    param["Initial concentration in negative electrode [mol.m-3]"] *= SOH
    param["Initial concentration in positive electrode [mol.m-3]"] *= SOH

    temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)
    resistance_factor = min(1.0, max(0.3, SOH * RESISTANCE_NOISE * temp_conductivity_multiplier))
    param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
    param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

    # The lever under test: reduce positive-electrode (NMC) solid-state
    # diffusivity. Chen2020 defines this as a plain constant; Mohtat2020
    # and OKane2022 define it as a callable (sto, T) -> value (itself a
    # constant under the hood for both, but exposed as a function) -- so
    # a divide-by-scalar only works for Chen2020 and must instead wrap
    # the callable for the other two, scaling its return value.
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


def simulate_one(label, base_params, diffusivity_factor, c_rate, ambient_c):
    try:
        param = build_param(base_params, diffusivity_factor, ambient_c)
    except Exception as err:
        return {"label": label, "diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED build_param ({type(err).__name__}): {err}"}
    current_a = c_rate * TARGET_AH
    experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {V_MIN} V"])

    sim1 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    try:
        sol1 = sim1.solve(initial_soc=INITIAL_SOC)
    except Exception as err:
        return {"label": label, "diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED default solve ({type(err).__name__}): {err}"}

    time_default = sol1["Time [s]"].entries
    voltage_default = sol1["Terminal voltage [V]"].entries
    capacity_default = sol1["Discharge capacity [A.h]"].entries
    if len(time_default) < 5:
        return {"label": label, "diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": "FAILED (too few default points)"}

    tf = time_default[-1]
    t_bound = tf * T_INTERP_SAFETY_FRACTION

    sim2 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    t_interp = np.linspace(0, t_bound, T_INTERP_N_POINTS)
    try:
        sol2 = sim2.solve(initial_soc=INITIAL_SOC, t_interp=t_interp)
    except Exception as err:
        return {"label": label, "diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
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
        return {"label": label, "diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": "COLLAPSED (0 target-zone points)"}

    zone_vals = dvdq[in_zone]
    n_outliers = int((np.abs(zone_vals) > OUTLIER_DVDQ_THRESHOLD).sum())

    # Outside-zone sanity check: mid-curve bins (3.0-4.0V) shouldn't be badly distorted.
    outside = (v_curr > TARGET_ZONE_MAX) & (v_curr <= 4.0)
    outside_vals = dvdq[outside]
    outside_outliers = int((np.abs(outside_vals) > OUTLIER_DVDQ_THRESHOLD).sum()) if outside.sum() else 0

    return {
        "label": label, "diffusivity_factor": diffusivity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
        "status": "OK", "n_zone": n_zone, "mean_dvdq": float(zone_vals.mean()),
        "min_dvdq": float(zone_vals.min()), "max_dvdq": float(zone_vals.max()),
        "n_outliers_zone": n_outliers, "n_outside_points": int(outside.sum()), "n_outliers_outside": outside_outliers,
    }


def main():
    results = []
    for nmc_set in NMC_PARAMETER_SETS:
        for factor in DIFFUSIVITY_FACTORS:
            for c_rate in C_RATE_VALUES:
                for ambient_c in AMBIENT_TEMP_VALUES:
                    r = simulate_one(nmc_set, param_nmc_bases[nmc_set], factor, c_rate, ambient_c)
                    results.append(r)
                    tag = "OK" if r["status"] == "OK" else r["status"]
                    print(f"{nmc_set:<12} factor={factor:<3} C-rate={c_rate:<4} T={ambient_c:<5} -> {tag}"
                          + (f"  mean_dvdq={r.get('mean_dvdq'):.2f}  n_zone={r.get('n_zone')}  "
                             f"outliers(zone/outside)={r.get('n_outliers_zone')}/{r.get('n_outliers_outside')}"
                             if r["status"] == "OK" else ""))

    df = pd.DataFrame(results)
    out_csv = "experiments/22_nmc_diffusivity_tuning/diagnostic_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to '{out_csv}'")

    print(f"\n{'=' * 90}\nSUMMARY: per parameter-set, per diffusivity factor -- robustness and magnitude\n{'=' * 90}")
    for nmc_set in NMC_PARAMETER_SETS:
        print(f"\n--- {nmc_set} ---")
        for factor in DIFFUSIVITY_FACTORS:
            subset = df[(df["label"] == nmc_set) & (df["diffusivity_factor"] == factor)]
            n_total = len(subset)
            n_failed = (subset["status"] != "OK").sum()
            ok = subset[subset["status"] == "OK"]
            n_target_hit = 0
            mean_mag = None
            n_outliers_total = 0
            if len(ok):
                n_target_hit = ((ok["mean_dvdq"] >= REAL_NMC_TARGET_MIN) & (ok["mean_dvdq"] <= REAL_NMC_TARGET_MAX)).sum()
                mean_mag = ok["mean_dvdq"].mean()
                n_outliers_total = ok["n_outliers_zone"].sum() + ok["n_outliers_outside"].sum()
            print(f"  factor={factor:<3}: {n_total - n_failed}/{n_total} robust (no failure/collapse) | "
                  f"{n_target_hit}/{len(ok)} land in real target (-8 to -3) | "
                  f"mean magnitude={mean_mag if mean_mag is None else round(mean_mag, 2)} | "
                  f"total outliers={n_outliers_total}")


if __name__ == "__main__":
    main()
