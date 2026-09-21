"""
Phase 1 diagnostic for experiment 16 ("cutoff hack"): before running a full
synthetic batch, check whether artificially lowering PyBaMM's simulated
discharge cutoff actually keeps the solver-termination artifact out of the
real-world target zone (2.5-3.0V, matching the real LFP/NMC data's own
observed floor -- see experiments/07_real_lfp_nmc_test/parse_real_lfp.py
and parse_real_nmc.py).

Background: root feature_engineering.py bins each raw voltage transition
using only its ENDING voltage (single bin, never split/smeared), so pushing
the true simulated cutoff down should move the whole artifact zone down
with it. But experiment 07's investigation found that for a minority of
NMC batteries (~13% of Mohtat2020's), MORE than one oversized solver step
occurs near the cutoff -- not just the final one -- contaminating dV/dQ up
to ~0.6V above the actual cutoff. Whether a much lower cutoff (untested
territory) produces similar, smaller, or larger contamination -- or
whether PyBaMM's SPM model even converges reliably that low -- has to be
checked directly, not assumed by analogy with the existing 1.8V/2.3V
cutoffs.

This script simulates a small batch (a few SOH values x each NMC parameter
set, LFP with Prada2013) at each of several candidate lower cutoffs, and
for every simulation:
  1. confirms PyBaMM solved without error/non-convergence,
  2. reports the last-sample voltage and the final jump size (same style
     of check as experiments/11_voltage_cutoff_comparison/RESULTS.md),
  3. flags any transition whose ENDING voltage lands in the 2.5-3.0V
     target zone but whose |dV/dQ| is an outlier (>50, comfortably above
     the largest genuine values seen anywhere in this project's real or
     artifact-free synthetic data -- see exp07's RESULTS.md real-vs-
     synthetic magnitude tables, real values never exceed ~10).

Does NOT modify simulate_batteries.py or feature_engineering.py -- this is
a standalone diagnostic reusing the same physics setup (parameter values,
capacity scaling, SOH resistance modeling) at a much smaller scale.

Usage: python3 experiments/16_low_voltage_only_sim_to_real/diagnose_lower_cutoff.py
"""
import pybamm
import numpy as np
import pandas as pd

CANDIDATE_CUTOFFS = [1.5, 1.0, 0.5]
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
OUTLIER_DVDQ_THRESHOLD = 50.0

NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
SOH_VALUES = [0.50, 0.675, 0.85]  # matches simulate_batteries.py's random.uniform(0.50, 0.85) range: low/mid/high
DISCHARGE_C_RATE = 0.6
TARGET_AH = 2.0  # mid capacity tier from simulate_batteries.py's CAPACITY_TARGETS_AH
INITIAL_SOC = 1.0  # full discharge, to guarantee the low-voltage tail is reached

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


def simulate_one(chem, label, base_params, soh, v_min):
    param = build_param(base_params, soh, TARGET_AH)
    current_a = DISCHARGE_C_RATE * TARGET_AH
    experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {v_min} V"])
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)

    try:
        sol = sim.solve(initial_soc=INITIAL_SOC)
    except pybamm.SolverError as err:
        sol = getattr(sim, "solution", None)
        if sol is None or len(sol.t) == 0:
            return {"chem": chem, "label": label, "soh": soh, "v_min": v_min,
                    "status": f"FAILED (SolverError, no data): {err}"}
        status = f"partial (SolverError, kept {len(sol.t)} points): {err}"
    except Exception as err:
        return {"chem": chem, "label": label, "soh": soh, "v_min": v_min,
                "status": f"FAILED ({type(err).__name__}): {err}"}
    else:
        status = "OK"

    voltage = sol["Terminal voltage [V]"].entries
    capacity = sol["Discharge capacity [A.h]"].entries

    if len(voltage) < 3:
        return {"chem": chem, "label": label, "soh": soh, "v_min": v_min, "status": f"{status} (too few points)"}

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

    return {
        "chem": chem, "label": label, "soh": soh, "v_min": v_min, "status": status,
        "last_v": round(float(last_v), 4), "final_jump_v": round(float(final_jump), 4),
        "n_points_in_zone": int(in_zone.sum()), "max_abs_dvdq_in_zone": max_abs_dvdq_in_zone,
        "n_outliers_in_zone": n_outliers_in_zone,
    }


def main():
    results = []
    for v_min in CANDIDATE_CUTOFFS:
        print(f"\n{'=' * 70}\nCandidate cutoff: {v_min} V\n{'=' * 70}")
        for soh in SOH_VALUES:
            print(f"\n-- LFP (Prada2013), SOH={soh} --")
            r = simulate_one("LFP", "Prada2013", param_lfp_base, soh, v_min)
            print(f"   {r}")
            results.append(r)

            for nmc_set in NMC_PARAMETER_SETS:
                print(f"\n-- NMC ({nmc_set}), SOH={soh} --")
                r = simulate_one("NMC", nmc_set, param_nmc_bases[nmc_set], soh, v_min)
                print(f"   {r}")
                results.append(r)

    df = pd.DataFrame(results)
    out_csv = "experiments/16_low_voltage_only_sim_to_real/diagnostic_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to '{out_csv}'")

    print(f"\n{'=' * 70}\nSUMMARY: contamination in the {TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V target zone, per cutoff\n{'=' * 70}")
    for v_min in CANDIDATE_CUTOFFS:
        subset = df[df["v_min"] == v_min]
        failed = subset[subset["status"].astype(str).str.startswith("FAILED")]
        ok = subset[~subset.index.isin(failed.index)]
        total_outliers = ok["n_outliers_in_zone"].sum() if "n_outliers_in_zone" in ok else 0
        max_dvdq = ok["max_abs_dvdq_in_zone"].max() if "max_abs_dvdq_in_zone" in ok and ok["max_abs_dvdq_in_zone"].notna().any() else None
        print(f"v_min={v_min}V: {len(failed)}/{len(subset)} solves failed | "
              f"outlier transitions in {TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V zone: {total_outliers} | "
              f"max |dV/dQ| in zone: {max_dvdq}")
        if len(failed):
            print(f"  Failures: {failed[['chem', 'label', 'soh', 'status']].to_dict('records')}")

    print("\nRecommendation: pick the LOWEST v_min above with 0 failed solves AND "
          "0 outlier transitions in the target zone. If none qualify, this approach "
          "needs a different candidate range or is not viable as proposed.")


if __name__ == "__main__":
    main()
