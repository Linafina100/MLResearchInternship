"""
Experiment 21, Phase 2 diagnostic: does directly softening LFP's OCP
function's tail rate constant fix its dV/dQ magnitude mismatch, where
diffusivity/kinetics tuning (Phase 1, diagnose_lfp_diffusivity_factor.py)
was proven to have zero effect?

LFP's OCP (Prada2013's parameter set actually borrows Afshar2017's LFP
fit, since Prada2013 itself doesn't define one):

    def LFP_ocp_Afshar2017(sto):
        c1 = -150 * sto
        c2 = -30 * (1 - sto)
        k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)
        return k

`sto` increases toward 1 as the cell discharges; near sto=1 this
evaluates to ~2.49V, matching the 2.5-3.0V target zone -- confirmed by
inspection, not assumed. The `-0.9*exp(-30*(1-sto))` term is what
produces the steep tail there: as sto->1, (1-sto)->0 and this term rises
sharply over a narrow stoichiometry window. This script softens the -30
rate constant (smaller magnitude spreads the same total voltage drop
over a wider stoichiometry range, smearing the tail) and sweeps
candidate values across the real C-rate (0.1-0.2)/temperature (15-35C)
grid at SOH=0.8, exactly as experiment 20's NMC diffusivity sweep and
this experiment's own Phase 1 did.

Explicitly an empirical calibration, not first-principles physics --
directly editing the fitted OCP function's shape, documented as such per
explicit agreement, not presented as more accurate electrochemistry.

Usage: python3 experiments/21_lfp_diffusivity_tuning/diagnose_lfp_ocp_rate_constant.py
"""
import pybamm
import numpy as np
import pandas as pd

V_MIN = 1.5
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
REAL_LFP_TARGET_MIN, REAL_LFP_TARGET_MAX = -2.9, -0.3  # experiment 20 Part A's own real-data measurement
OUTLIER_DVDQ_THRESHOLD = 50.0
T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6  # verified safe in experiments 18/20/21 -- do not widen this margin

# -30 = original (baseline, unmodified curve). Smaller magnitude = softer/smeared tail.
RATE_CONSTANTS = [-30, -25, -20, -15, -10, -7, -5, -4, -3, -2, -1]
C_RATE_VALUES = [0.1, 0.15, 0.2]
AMBIENT_TEMP_VALUES = [15.0, 25.0, 35.0]
TARGET_AH = 2.0
SOH = 0.8
INITIAL_SOC = 1.0
TEMP_RESISTANCE_COEFF = 0.02
RESISTANCE_NOISE = 1.0

model = pybamm.lithium_ion.SPM()
param_lfp_base = pybamm.ParameterValues("Prada2013")


def make_lfp_ocp(rate_constant):
    """Same functional form as LFP_ocp_Afshar2017, with the tail rate
    constant (-30 originally) replaced by `rate_constant`."""
    def lfp_ocp_softened(sto):
        c1 = -150 * sto
        c2 = rate_constant * (1 - sto)
        k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)
        return k
    return lfp_ocp_softened


def build_param(rate_constant, ambient_c):
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

    param["Positive electrode OCP [V]"] = make_lfp_ocp(rate_constant)
    return param


def transitions(time, voltage, capacity):
    order = np.argsort(time, kind="stable")
    t, v, q = time[order], voltage[order], capacity[order]
    dV, dQ = np.diff(v), np.diff(q)
    valid = dQ > 1e-5
    return v[1:][valid], (dV[valid] / dQ[valid])


def simulate_one(rate_constant, c_rate, ambient_c):
    try:
        param = build_param(rate_constant, ambient_c)
    except Exception as err:
        return {"rate_constant": rate_constant, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED build_param ({type(err).__name__}): {err}"}

    current_a = c_rate * TARGET_AH
    experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {V_MIN} V"])

    sim1 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    try:
        sol1 = sim1.solve(initial_soc=INITIAL_SOC)
    except Exception as err:
        return {"rate_constant": rate_constant, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED default solve ({type(err).__name__}): {err}"}

    time_default = sol1["Time [s]"].entries
    voltage_default = sol1["Terminal voltage [V]"].entries
    capacity_default = sol1["Discharge capacity [A.h]"].entries
    if len(time_default) < 5:
        return {"rate_constant": rate_constant, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": "FAILED (too few default points)"}

    tf = time_default[-1]
    t_bound = tf * T_INTERP_SAFETY_FRACTION

    sim2 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    t_interp = np.linspace(0, t_bound, T_INTERP_N_POINTS)
    try:
        sol2 = sim2.solve(initial_soc=INITIAL_SOC, t_interp=t_interp)
    except Exception as err:
        return {"rate_constant": rate_constant, "c_rate": c_rate, "ambient_c": ambient_c,
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
        return {"rate_constant": rate_constant, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": "COLLAPSED (0 target-zone points)"}

    zone_vals = dvdq[in_zone]
    n_outliers = int((np.abs(zone_vals) > OUTLIER_DVDQ_THRESHOLD).sum())

    # Outside-zone sanity check: does softening the tail distort the mid-curve plateau (3.0-4.0V)?
    outside = (v_curr > TARGET_ZONE_MAX) & (v_curr <= 4.0)
    outside_vals = dvdq[outside]
    outside_outliers = int((np.abs(outside_vals) > OUTLIER_DVDQ_THRESHOLD).sum()) if outside.sum() else 0

    return {
        "rate_constant": rate_constant, "c_rate": c_rate, "ambient_c": ambient_c,
        "status": "OK", "n_zone": n_zone, "mean_dvdq": float(zone_vals.mean()),
        "min_dvdq": float(zone_vals.min()), "max_dvdq": float(zone_vals.max()),
        "n_outliers_zone": n_outliers, "n_outside_points": int(outside.sum()), "n_outliers_outside": outside_outliers,
    }


def main():
    results = []
    for rc in RATE_CONSTANTS:
        for c_rate in C_RATE_VALUES:
            for ambient_c in AMBIENT_TEMP_VALUES:
                r = simulate_one(rc, c_rate, ambient_c)
                results.append(r)
                tag = "OK" if r["status"] == "OK" else r["status"]
                print(f"rate={rc:<4} C-rate={c_rate:<4} T={ambient_c:<5} -> {tag}"
                      + (f"  mean_dvdq={r.get('mean_dvdq'):.2f}  n_zone={r.get('n_zone')}  "
                         f"outliers(zone/outside)={r.get('n_outliers_zone')}/{r.get('n_outliers_outside')}"
                         if r["status"] == "OK" else ""))

    df = pd.DataFrame(results)
    out_csv = "experiments/21_lfp_diffusivity_tuning/diagnostic_results_ocp_rate.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to '{out_csv}'")

    print(f"\n{'=' * 90}\nSUMMARY: per rate constant -- robustness and magnitude (real target: -2.9 to -0.3)\n{'=' * 90}")
    for rc in RATE_CONSTANTS:
        subset = df[df["rate_constant"] == rc]
        n_total = len(subset)
        n_failed = (subset["status"] != "OK").sum()
        ok = subset[subset["status"] == "OK"]
        n_target_hit = 0
        mean_mag = None
        n_outliers_total = 0
        if len(ok):
            n_target_hit = ((ok["mean_dvdq"] >= REAL_LFP_TARGET_MIN) & (ok["mean_dvdq"] <= REAL_LFP_TARGET_MAX)).sum()
            mean_mag = ok["mean_dvdq"].mean()
            n_outliers_total = ok["n_outliers_zone"].sum() + ok["n_outliers_outside"].sum()
        print(f"  rate={rc:<4}: {n_total - n_failed}/{n_total} robust | "
              f"{n_target_hit}/{len(ok)} land in real target (-2.9 to -0.3) | "
              f"mean magnitude={mean_mag if mean_mag is None else round(mean_mag, 2)} | "
              f"total outliers={n_outliers_total}")


if __name__ == "__main__":
    main()
