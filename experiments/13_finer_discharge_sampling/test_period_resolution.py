"""
Tests the hypothesis that experiment 12's termination-step artifact
(dV/dQ exploding on each battery's final raw sample, right at the voltage
cutoff -- see experiments/11_voltage_cutoff_comparison/RESULTS.md for the
original discovery) is a sampling-resolution problem, fixable by forcing
PyBaMM to output the discharge trace on a finer, regular time grid instead
of relying on `simulate_batteries.py`'s current unset `period` (which lets
the solver report at its own adaptive/default spacing).

PyBaMM experiment steps accept a `(<duration> period)` suffix that sets
this output grid explicitly, e.g. `"Discharge at 1.2 A until 2.3 V (1
second period)"`. This script reruns the exact per-battery parameter
randomization `simulate_batteries.py` uses (chemistry pooling, capacity
scaling, SOH, resistance/temperature noise), restricted to the SOC 0.1-0.4
range where the artifact was worst, at several period settings, and
compares the final raw transition's dV/dQ across them -- without any full
re-simulation of a whole SOC sweep, since the batch result below is
decisive on its own.

Usage: python3 experiments/13_finer_discharge_sampling/test_period_resolution.py
"""
import warnings
warnings.filterwarnings("ignore")

import csv
import os
import random

import numpy as np
import pybamm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "period_resolution_results.csv")
RESULTS_COLUMNS = ["Period", "Chemistry", "Variation", "Last_dV", "Last_dQ", "Last_dVdQ"]

model = pybamm.lithium_ion.SPM()
param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}

LOWER_VOLTAGE_CUTOFF = {"LFP": 1.8, "NMC": 2.3}
TARGET_AH = 2.0
DISCHARGE_C_RATE = 0.6
N_VARIATIONS = 20
SOC_RANGE = (0.1, 0.4)  # experiment 03's worst-affected interval

# None = current simulate_batteries.py behaviour (no period set).
PERIOD_SETTINGS = [None, "10 seconds", "2 seconds", "1 second"]


def last_transition(chem, variation, period_str):
    """Reruns one (chem, variation)'s exact randomized parameters (same
    per-variation seed regardless of period, so period is the only thing
    that varies for a given variation number) and returns the final raw
    transition's (dV, dQ, dV/dQ)."""
    random.seed(1000 + variation)
    soc = random.uniform(*SOC_RANGE)
    soh = random.uniform(0.50, 0.85)
    ambient_c = random.uniform(0, 35)
    temp_mult = 1.0 + 0.02 * (ambient_c - 25.0)
    nmc_set_name = random.choice(NMC_PARAMETER_SETS)
    resistance_noise = random.uniform(0.8, 1.2)

    base = param_lfp_base if chem == "LFP" else param_nmc_bases[nmc_set_name]
    param = base.copy()
    base_cap = base["Nominal cell capacity [A.h]"]
    mult = TARGET_AH / base_cap
    param["Negative electrode thickness [m]"] *= mult
    param["Positive electrode thickness [m]"] *= mult
    param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
    param["Maximum concentration in positive electrode [mol.m-3]"] *= soh
    resistance_factor = min(1.0, max(0.3, soh * resistance_noise * temp_mult))
    param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
    param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

    current_a = DISCHARGE_C_RATE * TARGET_AH
    v_min = LOWER_VOLTAGE_CUTOFF[chem]
    text = f"Discharge at {current_a:.4f} A until {v_min} V"
    if period_str:
        text += f" ({period_str} period)"
    experiment = pybamm.Experiment([text])
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)

    try:
        sol = sim.solve(initial_soc=soc)
    except Exception:
        return None

    v = sol["Terminal voltage [V]"].entries
    q = sol["Discharge capacity [A.h]"].entries
    dv = np.diff(v)
    dq = np.diff(q)
    valid = dq > 1e-5
    if not valid.any():
        return None
    return float(dv[valid][-1]), float(dq[valid][-1]), float(dv[valid][-1] / dq[valid][-1])


def main():
    rows = []
    for period_str in PERIOD_SETTINGS:
        period_label = period_str or "default (no period)"
        print(f"\n{'=' * 60}\nPeriod: {period_label}\n{'=' * 60}")
        for chem in ("LFP", "NMC"):
            dvdqs = []
            for variation in range(N_VARIATIONS):
                result = last_transition(chem, variation, period_str)
                if result is None:
                    continue
                dv, dq, dvdq = result
                dvdqs.append(dvdq)
                rows.append({
                    "Period": period_label, "Chemistry": chem, "Variation": variation,
                    "Last_dV": dv, "Last_dQ": dq, "Last_dVdQ": dvdq,
                })
            arr = np.array(dvdqs)
            n_extreme = int((np.abs(arr) > 100).sum())
            print(f"  {chem}: n={len(arr)}, median dV/dQ={np.median(arr):.1f}, "
                  f"max|dV/dQ|={np.abs(arr).max():.1f}, n(|dV/dQ|>100)={n_extreme}/{len(arr)}")

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull per-battery results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
