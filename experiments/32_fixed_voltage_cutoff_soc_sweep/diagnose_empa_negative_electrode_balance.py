"""
Experiment 32: diagnoses whether shifting the NEGATIVE/POSITIVE electrode
CAPACITY BALANCE (not diffusivity, not the LFP OCP tail -- both already
ruled out, see this folder's RESULTS.md once written and the memory
entries [[exp29_lfp_ocp_tail_ceiling]] / [[exp29_negative_electrode_diffusivity_fix]])
can shift LFP's steep dV/dQ dive to a shallower, earlier voltage, closer
to real EMPA's observed ~2.6-2.7V knee (synthetic's equivalent steepness
currently only appears near 2.3-2.5V -- confirmed directly, see
RESULTS.md's "hypothesis 1" section).

WHY THIS LEVER: experiment 29's own diagnosis of the OCP-tail dead end
found the cell's terminal voltage crashes for reasons UNRELATED to LFP's
own OCP tail term -- "something else in the model (most likely the
negative electrode side) governs where this discharge trajectory ends."
That was only ever tested via negative-electrode DIFFUSIVITY (a
kinetics/rate parameter, ruled out -- caused a runaway escalation deeper
in discharge). This script tests a structurally different lever: the
negative electrode's CAPACITY relative to the positive electrode's,
independent of the thickness multiplier the project already uses to hit
a target pack Ah (which scales both electrodes by the SAME factor,
preserving their relative balance). Scaling the negative electrode's
capacity UP or DOWN relative to the positive electrode changes which
electrode's own stoichiometric limit governs when full-cell voltage
crashes -- a genuinely different mechanism from both already-ruled-out
levers.

METHODOLOGY LESSON APPLIED (from the negative-electrode-diffusivity dead
end): report the FULL dV/dQ-vs-voltage curve for every candidate, not
just one hand-picked zone's mean -- a narrow-zone-only check is exactly
what let that prior lever's runaway side effect go unnoticed until the
full evaluation.

Single-point full-discharge PyBaMM solve per candidate (SOH=0.8,
Initial_SOC=1.0, C-rate in {0.1,0.15,0.2} x ambient temp in {15,25,35}),
LFP only (NMC untouched) -- mirrors
experiments/22_29_sim_to_real_validation/29_snl_lfp_ocp_recalibration/diagnose_snl_lfp_ocp_rate_constant.py's
structure.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/diagnose_empa_negative_electrode_balance.py
"""
import os

import numpy as np
import pandas as pd
import pybamm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

V_MIN = 1.5  # natural/deep cutoff for the diagnostic -- lets us see the full curve shape, not just where EMPA happens to stop
TARGET_CROSSING_DVDQ = -4.0  # real EMPA LFP crosses this around 2.65V (interpolated between 2.7-2.6V's -4.10 and 2.8-2.7V's -2.98)
REAL_EMPA_CROSSING_V = 2.65  # see RESULTS.md's hypothesis-1 section for how this was measured
T_INTERP_N_POINTS = 3000
T_INTERP_SAFETY_FRACTION = 1 - 1e-6

# 1.0 = baseline (current project default, negative and positive electrode
# thickness scaled by the SAME factor). <1.0 shrinks the negative
# electrode's capacity relative to positive (should deplete SOONER);
# >1.0 grows it (should deplete LATER). Swept both directions since the
# direction that shifts the dive the right way isn't assumed in advance --
# the simulation decides, not a predicted formula.
NEG_CAPACITY_FACTORS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
C_RATE_VALUES = [0.1, 0.15, 0.2]
AMBIENT_TEMP_VALUES = [15.0, 25.0, 35.0]
TARGET_AH = 2.0  # matches EMPA's RESCALE_TARGET_AH
SOH = 0.8
INITIAL_SOC = 1.0
LFP_OCP_RATE_CONSTANT = -3  # already-established baseline (experiment 21), held fixed -- this diagnostic isolates the NEW lever only
TEMP_RESISTANCE_COEFF = 0.02
RESISTANCE_NOISE = 1.0

model = pybamm.lithium_ion.SPM()
param_lfp_base = pybamm.ParameterValues("Prada2013")


def make_lfp_ocp(rate_constant):
    def lfp_ocp_softened(sto):
        c1 = -150 * sto
        c2 = rate_constant * (1 - sto)
        k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)
        return k
    return lfp_ocp_softened


def build_param(neg_capacity_factor, ambient_c):
    param = param_lfp_base.copy()
    base_capacity_ah = param_lfp_base["Nominal cell capacity [A.h]"]
    mult = TARGET_AH / base_capacity_ah
    param["Negative electrode thickness [m]"] *= mult
    param["Positive electrode thickness [m]"] *= mult

    param["Maximum concentration in negative electrode [mol.m-3]"] *= SOH
    param["Maximum concentration in positive electrode [mol.m-3]"] *= SOH
    param["Initial concentration in negative electrode [mol.m-3]"] *= SOH
    param["Initial concentration in positive electrode [mol.m-3]"] *= SOH

    # THE LEVER UNDER TEST: scale negative electrode capacity independent
    # of positive electrode's, via its max/initial concentration together
    # (keeps the negative electrode's OWN initial stoichiometry fraction
    # unchanged -- only its total capacity changes).
    param["Maximum concentration in negative electrode [mol.m-3]"] *= neg_capacity_factor
    param["Initial concentration in negative electrode [mol.m-3]"] *= neg_capacity_factor

    temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)
    resistance_factor = min(1.0, max(0.3, SOH * RESISTANCE_NOISE * temp_conductivity_multiplier))
    param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
    param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

    param["Positive electrode OCP [V]"] = make_lfp_ocp(LFP_OCP_RATE_CONSTANT)
    return param


def transitions(time, voltage, capacity):
    order = np.argsort(time, kind="stable")
    t, v, q = time[order], voltage[order], capacity[order]
    dV, dQ = np.diff(v), np.diff(q)
    valid = dQ > 1e-5
    return v[1:][valid], (dV[valid] / dQ[valid])


def bin_dvdq(v_curr, dvdq, bin_width=0.1, v_min=1.5, v_max=3.6):
    """Full-range 0.1V-bin medians -- same convention as
    feature_engineering.py, computed directly here (no CSV round-trip
    needed) since this is a single-point diagnostic, not a dataset build."""
    edges = np.arange(v_min, v_max + bin_width, bin_width)
    bin_medians = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (v_curr >= lo) & (v_curr < hi)
        if mask.sum() >= 3:
            bin_medians[f"{hi:.1f}-{lo:.1f}"] = float(np.median(dvdq[mask]))
    return bin_medians


CROSSING_SEARCH_V_MAX = 3.0  # excludes the initial-transient dip right after
                              # discharge starts (observed directly: steep
                              # negative dV/dQ near 3.4-3.6V from the cell
                              # settling out of its start-of-discharge
                              # relaxation, unrelated to the end-of-discharge
                              # knee this diagnostic is actually about) --
                              # 3.0V sits safely below that transient and
                              # above every candidate's observed knee region


def find_crossing_voltage(v_curr, dvdq, threshold=TARGET_CROSSING_DVDQ):
    """Highest voltage BELOW CROSSING_SEARCH_V_MAX at which dV/dQ first
    reaches `threshold` (going more negative) during discharge -- the
    metric we're trying to shift from synthetic's current ~2.5V down
    toward real EMPA's ~2.65V. Restricted below CROSSING_SEARCH_V_MAX to
    avoid the initial-transient dip, which also crosses steep negative
    values but right at the START of discharge, not the end."""
    mask = v_curr <= CROSSING_SEARCH_V_MAX
    if not mask.any():
        return None
    v_sub, dvdq_sub = v_curr[mask], dvdq[mask]
    order = np.argsort(-v_sub)  # descending voltage = discharge order
    v_sorted, dvdq_sorted = v_sub[order], dvdq_sub[order]
    below = np.where(dvdq_sorted <= threshold)[0]
    if len(below) == 0:
        return None
    return float(v_sorted[below[0]])


def simulate_one(neg_capacity_factor, c_rate, ambient_c):
    try:
        param = build_param(neg_capacity_factor, ambient_c)
    except Exception as err:
        return {"neg_capacity_factor": neg_capacity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED build_param ({type(err).__name__}): {err}"}

    current_a = c_rate * TARGET_AH
    experiment = pybamm.Experiment([f"Discharge at {current_a:.4f} A until {V_MIN} V"])

    sim1 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    try:
        sol1 = sim1.solve(initial_soc=INITIAL_SOC)
    except Exception as err:
        return {"neg_capacity_factor": neg_capacity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED default solve ({type(err).__name__}): {err}"}

    time_default = sol1["Time [s]"].entries
    voltage_default = sol1["Terminal voltage [V]"].entries
    capacity_default = sol1["Discharge capacity [A.h]"].entries
    if len(time_default) < 5:
        return {"neg_capacity_factor": neg_capacity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": "FAILED (too few default points)"}

    tf = time_default[-1]
    t_bound = tf * T_INTERP_SAFETY_FRACTION

    sim2 = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    t_interp = np.linspace(0, t_bound, T_INTERP_N_POINTS)
    try:
        sol2 = sim2.solve(initial_soc=INITIAL_SOC, t_interp=t_interp)
    except Exception as err:
        return {"neg_capacity_factor": neg_capacity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
                "status": f"FAILED t_interp solve ({type(err).__name__}): {err}"}

    time_dense = sol2["Time [s]"].entries
    voltage_dense = sol2["Terminal voltage [V]"].entries
    capacity_dense = sol2["Discharge capacity [A.h]"].entries

    after_mask = time_default > time_dense[-1]
    time_spliced = np.concatenate([time_dense, time_default[after_mask]])
    voltage_spliced = np.concatenate([voltage_dense, voltage_default[after_mask]])
    capacity_spliced = np.concatenate([capacity_dense, capacity_default[after_mask]])

    v_curr, dvdq = transitions(time_spliced, voltage_spliced, capacity_spliced)
    bin_medians = bin_dvdq(v_curr, dvdq)
    crossing_v = find_crossing_voltage(v_curr, dvdq)
    final_capacity = float(capacity_spliced[-1])
    final_voltage = float(voltage_spliced[-1])

    return {
        "neg_capacity_factor": neg_capacity_factor, "c_rate": c_rate, "ambient_c": ambient_c,
        "status": "OK", "crossing_voltage": crossing_v, "final_capacity_ah": final_capacity,
        "final_voltage": final_voltage, "bin_medians": bin_medians,
    }


def main():
    print(f"Target: real EMPA LFP crosses dV/dQ={TARGET_CROSSING_DVDQ} around {REAL_EMPA_CROSSING_V}V "
          f"(synthetic baseline, factor=1.0, currently crosses much deeper -- see below)\n")

    results = []
    for factor in NEG_CAPACITY_FACTORS:
        for c_rate in C_RATE_VALUES:
            for ambient_c in AMBIENT_TEMP_VALUES:
                r = simulate_one(factor, c_rate, ambient_c)
                results.append(r)
                tag = r["status"]
                if tag == "OK":
                    cv = r["crossing_voltage"]
                    cv_str = f"{cv:.3f}V" if cv is not None else "NEVER REACHED"
                    print(f"factor={factor:<5} C-rate={c_rate:<4} T={ambient_c:<5} -> crossing_V={cv_str:<14} "
                          f"final_cap={r['final_capacity_ah']:.3f}Ah final_V={r['final_voltage']:.3f}")
                else:
                    print(f"factor={factor:<5} C-rate={c_rate:<4} T={ambient_c:<5} -> {tag}")

    # Flatten bin_medians into columns for the saved CSV (full range, per the lesson learned)
    all_bin_keys = sorted({k for r in results if r.get("bin_medians") for k in r["bin_medians"]},
                           key=lambda b: -float(b.split("-")[0]))
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "bin_medians"}
        for bk in all_bin_keys:
            row[f"dvdq_{bk}"] = r.get("bin_medians", {}).get(bk)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(SCRIPT_DIR, "diagnostic_results_empa_negative_electrode_balance.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nFull results (including every surviving bin's median, not just the crossing-voltage summary) "
          f"saved to '{out_csv}'")

    print(f"\n{'=' * 100}\nSUMMARY per factor: median crossing voltage, and full-range bin medians "
          f"(watch for runaway escalation in the deepest bins -- the exact failure mode the "
          f"negative-electrode-diffusivity attempt missed by only checking one zone)\n{'=' * 100}")
    for factor in NEG_CAPACITY_FACTORS:
        subset = df[(df["neg_capacity_factor"] == factor) & (df["status"] == "OK")]
        if len(subset) == 0:
            print(f"  factor={factor}: no successful solves")
            continue
        crossings = subset["crossing_voltage"].dropna()
        cv_med = crossings.median() if len(crossings) else None
        n_never = subset["crossing_voltage"].isna().sum()
        print(f"\n  factor={factor}: median crossing_V={cv_med if cv_med is None else round(cv_med, 3)} "
              f"({n_never}/{len(subset)} never reached {TARGET_CROSSING_DVDQ}), "
              f"median final_capacity={subset['final_capacity_ah'].median():.3f}Ah")
        bin_cols = [c for c in subset.columns if c.startswith("dvdq_")]
        medians = subset[bin_cols].median().dropna()
        print("   " + "  ".join(f"{c.replace('dvdq_', '')}={v:.2f}" for c, v in medians.items()))


if __name__ == "__main__":
    main()
