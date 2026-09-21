"""
Quick re-verification of experiment 16's diagnostic (../16_low_voltage_only_sim_to_real/diagnose_lower_cutoff.py)
under this experiment's very different conditions: SOH 0.8-1.0 (vs. 0.50-0.85)
and C-rate 0.1-0.2 (vs. a fixed 0.6C). Solver step dynamics near the
voltage cutoff can depend on discharge rate, so the 1.5V cutoff's safety
margin (no artifact contamination in the 2.5-3.0V target zone) is
re-checked here rather than assumed to carry over unchanged.

Usage: python3 experiments/17_high_soh_low_crate_sim_to_real/diagnose_cutoff_at_low_crate.py
"""
import pybamm
import numpy as np
import pandas as pd

V_MIN = 1.5
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
OUTLIER_DVDQ_THRESHOLD = 50.0

NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
SOH_VALUES = [0.80, 0.90, 1.00]
C_RATE_VALUES = [0.10, 0.15, 0.20]
TARGET_AH = 2.0
INITIAL_SOC = 1.0

model = pybamm.lithium_ion.SPM()
param_lfp_base = pybamm.ParameterValues("Prada2013")
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


def simulate_one(chem, label, base_params, soh, c_rate):
    param = build_param(base_params, soh, TARGET_AH)
    current_a = c_rate * TARGET_AH
    experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {V_MIN} V"])
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)

    try:
        sol = sim.solve(initial_soc=INITIAL_SOC)
    except pybamm.SolverError as err:
        sol = getattr(sim, "solution", None)
        if sol is None or len(sol.t) == 0:
            return {"chem": chem, "label": label, "soh": soh, "c_rate": c_rate,
                    "status": f"FAILED (SolverError, no data): {err}"}
        status = f"partial (SolverError, kept {len(sol.t)} points): {err}"
    except Exception as err:
        return {"chem": chem, "label": label, "soh": soh, "c_rate": c_rate,
                "status": f"FAILED ({type(err).__name__}): {err}"}
    else:
        status = "OK"

    voltage = sol["Terminal voltage [V]"].entries
    capacity = sol["Discharge capacity [A.h]"].entries

    if len(voltage) < 3:
        return {"chem": chem, "label": label, "soh": soh, "c_rate": c_rate, "status": f"{status} (too few points)"}

    last_v, second_last_v = voltage[-1], voltage[-2]
    final_jump = second_last_v - last_v

    dV = np.diff(voltage)
    dQ = np.diff(capacity)
    valid = dQ > 1e-5
    v_curr = voltage[1:]

    in_zone = valid & (v_curr >= TARGET_ZONE_MIN) & (v_curr < TARGET_ZONE_MAX)
    zone_dvdq = (dV[in_zone] / dQ[in_zone]) if in_zone.any() else np.array([])
    max_abs_dvdq_in_zone = float(np.max(np.abs(zone_dvdq))) if len(zone_dvdq) else None
    n_outliers_in_zone = int((np.abs(zone_dvdq) > OUTLIER_DVDQ_THRESHOLD).sum()) if len(zone_dvdq) else 0
    n_points_in_zone = int(in_zone.sum())

    return {
        "chem": chem, "label": label, "soh": soh, "c_rate": c_rate, "status": status,
        "last_v": round(float(last_v), 4), "final_jump_v": round(float(final_jump), 4),
        "n_points_in_zone": n_points_in_zone, "max_abs_dvdq_in_zone": max_abs_dvdq_in_zone,
        "n_outliers_in_zone": n_outliers_in_zone,
    }


def main():
    results = []
    for c_rate in C_RATE_VALUES:
        print(f"\n{'=' * 70}\nC-rate: {c_rate}\n{'=' * 70}")
        for soh in SOH_VALUES:
            print(f"\n-- LFP (Prada2013), SOH={soh} --")
            r = simulate_one("LFP", "Prada2013", param_lfp_base, soh, c_rate)
            print(f"   {r}")
            results.append(r)

            for nmc_set in NMC_PARAMETER_SETS:
                print(f"\n-- NMC ({nmc_set}), SOH={soh} --")
                r = simulate_one("NMC", nmc_set, param_nmc_bases[nmc_set], soh, c_rate)
                print(f"   {r}")
                results.append(r)

    df = pd.DataFrame(results)
    out_csv = "experiments/17_high_soh_low_crate_sim_to_real/diagnostic_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to '{out_csv}'")

    print(f"\n{'=' * 70}\nSUMMARY: contamination in the {TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V target zone, per C-rate\n{'=' * 70}")
    for c_rate in C_RATE_VALUES:
        subset = df[df["c_rate"] == c_rate]
        failed = subset[subset["status"].astype(str).str.startswith("FAILED")]
        ok = subset[~subset.index.isin(failed.index)]
        total_outliers = ok["n_outliers_in_zone"].sum() if "n_outliers_in_zone" in ok else 0
        max_dvdq = ok["max_abs_dvdq_in_zone"].max() if "max_abs_dvdq_in_zone" in ok and ok["max_abs_dvdq_in_zone"].notna().any() else None
        total_pts = ok["n_points_in_zone"].sum() if "n_points_in_zone" in ok else 0
        print(f"c_rate={c_rate}: {len(failed)}/{len(subset)} solves failed | "
              f"points in {TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V zone: {total_pts} | "
              f"outlier transitions: {total_outliers} | max |dV/dQ| in zone: {max_dvdq}")
        if len(failed):
            print(f"  Failures: {failed[['chem', 'label', 'soh', 'status']].to_dict('records')}")

    print(f"\nAt V_min={V_MIN}V: safe if 0 outliers and 0 failures across all C-rates tested.")


if __name__ == "__main__":
    main()
