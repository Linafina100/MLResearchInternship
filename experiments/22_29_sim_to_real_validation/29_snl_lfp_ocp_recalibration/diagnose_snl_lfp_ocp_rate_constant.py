"""
Experiment 29: re-calibrate LFP's OCP tail rate constant against SNL's
own real cells, mirroring experiment 21's exact methodology
(diagnose_lfp_ocp_rate_constant.py, which calibrated rate=-3 against
experiment 07's real LFP cell) but with two changes:

1. REAL_LFP_TARGET_MIN/MAX is measured directly from SNL's own real data
   (data/27_real_snl_soc_sweep/raw/real_snl_soc_sweep_raw.csv,
   Initial_SOC==1.0 slice -- untruncated full discharge, the same set
   used for the exp24 SNL check), over the same 2.5-3.0V target zone
   experiment 21 used -- not experiment 07's -2.9/-0.3 figure. SNL's real
   LFP target-zone mean (mean-of-5-per-bin-means, same convention exp21
   used for exp07) is -6.65 -- roughly 5x steeper than exp07's -1.28.
2. RATE_CONSTANTS extended toward steeper (more negative) values than the
   original -30 baseline, since even -30 (unmodified) only reaches -3.37
   in exp21's own sweep -- SNL needs something steeper than baseline, not
   a softer value like exp21 found for exp07.

Same single-point full-discharge PyBaMM solve per candidate (SOH=0.8,
Initial_SOC=1.0, C-rate in {0.1,0.15,0.2} x ambient temp in {15,25,35}),
same softened-OCP functional form. Explicitly an empirical calibration,
not first-principles physics -- same caveat as experiment 21.

Usage: python3 experiments/22_29_sim_to_real_validation/29_snl_lfp_ocp_recalibration/diagnose_snl_lfp_ocp_rate_constant.py
"""
import os

import numpy as np
import pandas as pd
import pybamm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SNL_SOC_SWEEP_CSV = os.path.join(PROJECT_DIR, "data", "27_real_snl_soc_sweep", "raw", "real_snl_soc_sweep_raw.csv")

V_MIN = 1.5
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
OUTLIER_DVDQ_THRESHOLD = 50.0
T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6

# -30 = experiment 21/26's current baseline (calibrated against exp07, not SNL).
# Softer values (toward 0) were experiment 21's own search direction for exp07.
# Steeper values (more negative than -30) are this experiment's search direction,
# since SNL's real magnitude is far larger than even the unmodified baseline reaches.
RATE_CONSTANTS = [-30, -50, -75, -100, -150, -200, -10, -3]  # baseline + steeper sweep, plus exp07/exp26's two prior calibration points for reference
C_RATE_VALUES = [0.1, 0.15, 0.2]
AMBIENT_TEMP_VALUES = [15.0, 25.0, 35.0]
TARGET_AH = 2.0
SOH = 0.8
INITIAL_SOC = 1.0
TEMP_RESISTANCE_COEFF = 0.02
RESISTANCE_NOISE = 1.0

model = pybamm.lithium_ion.SPM()
param_lfp_base = pybamm.ParameterValues("Prada2013")


def measure_real_snl_target():
    """Mirrors exp21's own real-target computation: mean of the 5
    target-zone bins' per-bin means, computed directly from real SNL LFP
    data (Initial_SOC==1.0, untruncated full discharge)."""
    import sys
    sys.path.insert(0, PROJECT_DIR)
    from feature_engineering import create_features_by_voltage_bins

    real_df = pd.read_csv(SNL_SOC_SWEEP_CSV, low_memory=False)
    real_df = real_df[real_df["Initial_SOC"] == 1.0].copy()
    tmp_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_snl_real_only.csv")
    real_df.to_csv(tmp_csv, index=False)

    features_df = create_features_by_voltage_bins(
        tmp_csv, output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_features"),
        exclude_final_transition=True, min_chemistry_coverage=0.0,
    )
    os.remove(tmp_csv)

    zone_bins = [f"dV_dQ_V_{TARGET_ZONE_MIN + 0.1*(i+1):.1f}_{TARGET_ZONE_MIN + 0.1*i:.1f}" for i in range(int(round((TARGET_ZONE_MAX-TARGET_ZONE_MIN)/0.1)))]
    zone_bins = [b for b in zone_bins if b in features_df.columns]
    real_lfp = features_df[features_df["Chemistry"] == "LFP"]
    per_bin_means = real_lfp[zone_bins].mean()
    print(f"Real SNL LFP target-zone per-bin means ({zone_bins}):")
    print(per_bin_means.to_string())
    target_mean = per_bin_means.mean()
    print(f"Target-zone mean (mean of per-bin means, exp21 convention): {target_mean:.3f}\n")
    return target_mean


def make_lfp_ocp(rate_constant, coefficient=-0.9):
    """coefficient is the -0.9 multiplier on the tail term -- at sto=1
    (end of discharge) this term alone sets k(1) = 3.387 + coefficient,
    so its magnitude directly controls how far voltage drops at the very
    end of discharge, independent of how fast (rate_constant) it gets
    there. -0.9 is Afshar2017's original fitted value."""
    def lfp_ocp_softened(sto):
        c1 = -150 * sto
        c2 = rate_constant * (1 - sto)
        k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) + coefficient * np.exp(c2)
        return k
    return lfp_ocp_softened


def build_param(rate_constant, ambient_c, coefficient=-0.9):
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

    param["Positive electrode OCP [V]"] = make_lfp_ocp(rate_constant, coefficient)
    return param


def transitions(time, voltage, capacity):
    order = np.argsort(time, kind="stable")
    t, v, q = time[order], voltage[order], capacity[order]
    dV, dQ = np.diff(v), np.diff(q)
    valid = dQ > 1e-5
    return v[1:][valid], (dV[valid] / dQ[valid])


def simulate_one(rate_constant, c_rate, ambient_c, coefficient=-0.9):
    try:
        param = build_param(rate_constant, ambient_c, coefficient)
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

    outside = (v_curr > TARGET_ZONE_MAX) & (v_curr <= 4.0)
    outside_vals = dvdq[outside]
    outside_outliers = int((np.abs(outside_vals) > OUTLIER_DVDQ_THRESHOLD).sum()) if outside.sum() else 0

    return {
        "rate_constant": rate_constant, "c_rate": c_rate, "ambient_c": ambient_c, "coefficient": coefficient,
        "status": "OK", "n_zone": n_zone, "mean_dvdq": float(zone_vals.mean()),
        "min_dvdq": float(zone_vals.min()), "max_dvdq": float(zone_vals.max()),
        "n_outliers_zone": n_outliers, "n_outside_points": int(outside.sum()), "n_outliers_outside": outside_outliers,
    }


# Phase 2, discovered mid-run: the rate constant SATURATES (see RESULTS.md
# -- every value from -30 to -200 gives the identical ~-3.4 mean), because
# exp(rate*(1-sto)) collapses to 0 away from sto=1 regardless of how
# negative rate is -- it only controls how NARROW the transition is, not
# how DEEP. The -0.9 coefficient on that same term is what sets the
# voltage drop's actual depth (k(sto=1) = 3.387 + coefficient) -- still
# the same OCP tail term, the other free parameter in it.
COEFFICIENT_VALUES = [-0.9, -1.2, -1.5, -1.8, -2.1, -2.4]
COEFFICIENT_SWEEP_RATE = -30  # rate barely matters once >~10 in magnitude; hold at the original baseline


def main():
    real_target = measure_real_snl_target()
    target_min, target_max = real_target * 1.15, real_target * 0.85  # +/-15% band around the measured mean

    results = []
    print(f"{'=' * 90}\nPhase 1: rate-constant sweep (coefficient held at Afshar2017's original -0.9)\n{'=' * 90}")
    for rc in RATE_CONSTANTS:
        for c_rate in C_RATE_VALUES:
            for ambient_c in AMBIENT_TEMP_VALUES:
                r = simulate_one(rc, c_rate, ambient_c)
                results.append(r)
                tag = "OK" if r["status"] == "OK" else r["status"]
                print(f"rate={rc:<5} C-rate={c_rate:<4} T={ambient_c:<5} -> {tag}"
                      + (f"  mean_dvdq={r.get('mean_dvdq'):.2f}  n_zone={r.get('n_zone')}  "
                         f"outliers(zone/outside)={r.get('n_outliers_zone')}/{r.get('n_outliers_outside')}"
                         if r["status"] == "OK" else ""))

    print(f"\n{'=' * 90}\nPhase 2: coefficient sweep (rate constant held at {COEFFICIENT_SWEEP_RATE})\n{'=' * 90}")
    for coef in COEFFICIENT_VALUES:
        for c_rate in C_RATE_VALUES:
            for ambient_c in AMBIENT_TEMP_VALUES:
                r = simulate_one(COEFFICIENT_SWEEP_RATE, c_rate, ambient_c, coefficient=coef)
                results.append(r)
                tag = "OK" if r["status"] == "OK" else r["status"]
                print(f"coef={coef:<5} C-rate={c_rate:<4} T={ambient_c:<5} -> {tag}"
                      + (f"  mean_dvdq={r.get('mean_dvdq'):.2f}  n_zone={r.get('n_zone')}  "
                         f"outliers(zone/outside)={r.get('n_outliers_zone')}/{r.get('n_outliers_outside')}"
                         if r["status"] == "OK" else ""))

    df = pd.DataFrame(results)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_results_snl_ocp_rate.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to '{out_csv}'")

    print(f"\n{'=' * 90}\nSUMMARY: Phase 1, per rate constant "
          f"(real SNL target: {target_min:.2f} to {target_max:.2f}, measured mean {real_target:.2f})\n{'=' * 90}")
    for rc in RATE_CONSTANTS:
        subset = df[(df["rate_constant"] == rc) & (df["coefficient"] == -0.9)]
        n_total = len(subset)
        n_failed = (subset["status"] != "OK").sum()
        ok = subset[subset["status"] == "OK"]
        n_target_hit = 0
        mean_mag = None
        n_outliers_total = 0
        if len(ok):
            n_target_hit = ((ok["mean_dvdq"] >= target_min) & (ok["mean_dvdq"] <= target_max)).sum()
            mean_mag = ok["mean_dvdq"].mean()
            n_outliers_total = ok["n_outliers_zone"].sum() + ok["n_outliers_outside"].sum()
        print(f"  rate={rc:<5}: {n_total - n_failed}/{n_total} robust | "
              f"{n_target_hit}/{len(ok)} land in real SNL target ({target_min:.2f} to {target_max:.2f}) | "
              f"mean magnitude={mean_mag if mean_mag is None else round(mean_mag, 2)} | "
              f"total outliers={n_outliers_total}")

    print(f"\n{'=' * 90}\nSUMMARY: Phase 2, per coefficient (rate={COEFFICIENT_SWEEP_RATE}) "
          f"(real SNL target: {target_min:.2f} to {target_max:.2f}, measured mean {real_target:.2f})\n{'=' * 90}")
    for coef in COEFFICIENT_VALUES:
        subset = df[(df["rate_constant"] == COEFFICIENT_SWEEP_RATE) & (df["coefficient"] == coef)]
        n_total = len(subset)
        n_failed = (subset["status"] != "OK").sum()
        ok = subset[subset["status"] == "OK"]
        n_target_hit = 0
        mean_mag = None
        n_outliers_total = 0
        if len(ok):
            n_target_hit = ((ok["mean_dvdq"] >= target_min) & (ok["mean_dvdq"] <= target_max)).sum()
            mean_mag = ok["mean_dvdq"].mean()
            n_outliers_total = ok["n_outliers_zone"].sum() + ok["n_outliers_outside"].sum()
        print(f"  coef={coef:<5}: {n_total - n_failed}/{n_total} robust | "
              f"{n_target_hit}/{len(ok)} land in real SNL target ({target_min:.2f} to {target_max:.2f}) | "
              f"mean magnitude={mean_mag if mean_mag is None else round(mean_mag, 2)} | "
              f"total outliers={n_outliers_total}")


if __name__ == "__main__":
    main()
