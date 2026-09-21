"""
Follow-up to diagnose_t_interp_density.py: that diagnostic found t_interp
only cleanly recovers synthetic NMC's missing 2.5-3.0V samples for
undegraded (SOH=1.0) Chen2020/OKane2022 cells, and that Mohtat2020's
gains (at every SOH tested) come with real outlier contamination. Tested
SOH 0.6-1.0 and C-rate 0.2-1.0.

Hypothesis under test here: the physical near-discontinuity found at
lower SOH is a function of *severe* degradation and/or *high* current
specifically -- so restricting to milder, more plausible recycling
conditions might avoid it. Tests SOH 0.8-0.95, C-rate 0.1-0.2, and
(newly) ambient temperature 15-35C, which the original diagnostic didn't
model at all.

Reuses every function from diagnose_t_interp_density.py unchanged
(transitions, zone_stats, the same solve-twice-and-splice logic in
simulate_one, and the T_INTERP_SAFETY_FRACTION = 1 - 1e-6 bound --
experiment 18's own fix for a 95%-margin bug that silently produced
false negatives; not to be reintroduced). Kept as a separate script
rather than overwriting the original, so experiment 18's committed
27-combo result stays exactly reproducible from its own file.

Adds temperature-driven resistance modeling, absent from the original
diagnostic, reusing the exact pattern from
experiments/17_high_soh_low_crate_sim_to_real/simulate_batteries_high_soh_low_crate.py
(lines ~139-171): ambient temperature shifts a conductivity multiplier,
combined with SOH and a resistance-noise factor into a single
"resistance_factor" applied to both electrodes' conductivity.
resistance_noise is held fixed at 1.0 here (vs. the full pipeline's
per-battery random.uniform(0.8, 1.2)) for clean, reproducible pairwise
comparisons across this grid -- a deliberate simplification.

Usage: python3 experiments/18_dense_interp_nmc_sampling/diagnose_t_interp_density_mild_conditions.py
"""
import pybamm
import numpy as np
import pandas as pd

V_MIN = 1.5
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
OUTLIER_DVDQ_THRESHOLD = 50.0
T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6  # see diagnose_t_interp_density.py -- do not use a large margin

NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
SOH_VALUES = [0.80, 0.875, 0.95]
C_RATE_VALUES = [0.1, 0.2]
AMBIENT_TEMP_VALUES = [15.0, 25.0, 35.0]
TARGET_AH = 2.0
INITIAL_SOC = 1.0

TEMP_RESISTANCE_COEFF = 0.02
RESISTANCE_NOISE = 1.0  # fixed (no per-battery randomness) for clean pairwise comparison

model = pybamm.lithium_ion.SPM()
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}


def build_param(base_params, soh, target_ah, ambient_c):
    param = base_params.copy()
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    mult = target_ah / base_capacity_ah
    param["Negative electrode thickness [m]"] *= mult
    param["Positive electrode thickness [m]"] *= mult
    param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
    param["Maximum concentration in positive electrode [mol.m-3]"] *= soh

    temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)
    resistance_factor = min(1.0, max(0.3, soh * RESISTANCE_NOISE * temp_conductivity_multiplier))
    param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
    param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

    return param, resistance_factor


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


def simulate_one(label, base_params, soh, c_rate, ambient_c):
    param, resistance_factor = build_param(base_params, soh, TARGET_AH, ambient_c)
    current_a = c_rate * TARGET_AH
    experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {V_MIN} V"])

    sim1 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    try:
        sol1 = sim1.solve(initial_soc=INITIAL_SOC)
    except pybamm.SolverError as err:
        sol1 = getattr(sim1, "solution", None)
        if sol1 is None or len(sol1.t) == 0:
            return {"label": label, "soh": soh, "c_rate": c_rate, "ambient_c": ambient_c,
                    "status": f"FAILED (SolverError, no data): {err}"}
    except Exception as err:
        return {"label": label, "soh": soh, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED ({type(err).__name__}): {err}"}

    time_default = sol1["Time [s]"].entries
    voltage_default = sol1["Terminal voltage [V]"].entries
    capacity_default = sol1["Discharge capacity [A.h]"].entries
    if len(time_default) < 5:
        return {"label": label, "soh": soh, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": "FAILED (too few default points)"}

    tf = time_default[-1]
    t_bound = tf * T_INTERP_SAFETY_FRACTION

    sim2 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    t_interp = np.linspace(0, t_bound, T_INTERP_N_POINTS)
    try:
        sol2 = sim2.solve(initial_soc=INITIAL_SOC, t_interp=t_interp)
    except Exception as err:
        return {"label": label, "soh": soh, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED (t_interp solve, {type(err).__name__}): {err}"}

    time_dense = sol2["Time [s]"].entries
    voltage_dense = sol2["Terminal voltage [V]"].entries
    capacity_dense = sol2["Discharge capacity [A.h]"].entries

    after_mask = time_default > time_dense[-1]
    time_spliced = np.concatenate([time_dense, time_default[after_mask]])
    voltage_spliced = np.concatenate([voltage_dense, voltage_default[after_mask]])
    capacity_spliced = np.concatenate([capacity_dense, capacity_default[after_mask]])

    v_curr_default, dvdq_default = transitions(time_default, voltage_default, capacity_default)
    v_curr_spliced, dvdq_spliced = transitions(time_spliced, voltage_spliced, capacity_spliced)

    n_zone_default, max_dvdq_default, n_outliers_default = zone_stats(v_curr_default, dvdq_default)
    n_zone_spliced, max_dvdq_spliced, n_outliers_spliced = zone_stats(v_curr_spliced, dvdq_spliced)

    seam_dvdq = None
    if after_mask.sum() > 0:
        dV_seam = voltage_default[after_mask][0] - voltage_dense[-1]
        dQ_seam = capacity_default[after_mask][0] - capacity_dense[-1]
        seam_dvdq = float(dV_seam / dQ_seam) if dQ_seam > 1e-9 else None

    return {
        "label": label, "soh": soh, "c_rate": c_rate, "ambient_c": ambient_c,
        "resistance_factor": round(resistance_factor, 4), "status": "OK",
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
                for ambient_c in AMBIENT_TEMP_VALUES:
                    print(f"\n-- {nmc_set}, SOH={soh}, C-rate={c_rate}, Ambient={ambient_c}C --")
                    r = simulate_one(nmc_set, param_nmc_bases[nmc_set], soh, c_rate, ambient_c)
                    print(f"   {r}")
                    results.append(r)

    df = pd.DataFrame(results)
    out_csv = "experiments/18_dense_interp_nmc_sampling/diagnostic_results_mild_conditions.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to '{out_csv}'")

    ok = df[df["status"] == "OK"]
    failed = df[df["status"] != "OK"]
    print(f"\n{'=' * 70}\nSUMMARY ({len(ok)}/{len(df)} solves succeeded)\n{'=' * 70}")
    if len(failed):
        print("Failures:")
        print(failed[["label", "soh", "c_rate", "ambient_c", "status"]].to_string(index=False))

    print(f"\nTotal target-zone (2.5-3.0V) points -- default: {ok['n_zone_default'].sum()}, "
          f"spliced: {ok['n_zone_spliced'].sum()}")
    print(f"Combos with >=1 genuine target-zone point -- default: {(ok['n_zone_default'] > 0).sum()}/{len(ok)}, "
          f"spliced: {(ok['n_zone_spliced'] > 0).sum()}/{len(ok)}")
    print(f"Outlier transitions in target zone -- default: {ok['n_outliers_zone_default'].sum()}, "
          f"spliced: {ok['n_outliers_zone_spliced'].sum()}")
    max_spliced = ok['max_dvdq_zone_spliced'].max()
    print(f"Max |dV/dQ| in target zone -- default: {ok['max_dvdq_zone_default'].max()}, "
          f"spliced: {max_spliced}")

    print("\nPer-parameter-set breakdown (spliced target-zone points, outliers):")
    print(ok.groupby("label")[["n_zone_spliced", "n_outliers_zone_spliced"]].sum())

    print("\nPer-SOH breakdown (spliced target-zone points, outliers):")
    print(ok.groupby("soh")[["n_zone_spliced", "n_outliers_zone_spliced"]].sum())

    print("\nPer-temperature breakdown (spliced target-zone points, outliers):")
    print(ok.groupby("ambient_c")[["n_zone_spliced", "n_outliers_zone_spliced"]].sum())


if __name__ == "__main__":
    main()
