import os
import pybamm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Fixed seed so a generation run (including which random SOH/size draws, if
# any, fail to solve) is reproducible run-to-run instead of silently varying
# every time, which previously made sample-count deficits impossible to
# investigate after the fact.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



# All generated data lives under data/<run_label>/<kind>/, so different runs
# (the default standalone run vs. each run_soc_sweep.py interval) never
# overwrite each other and files are easy to find by what produced them and
# what they contain. RUN_LABEL defaults to "default" for a plain standalone
# run; run_soc_sweep.py sets it per SOC interval (e.g. "soc_0.1-0.4").
DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "default")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

# Individual paths remain overridable so callers can pin an exact location
# if ever needed, but default to the sorted-by-run/sorted-by-kind layout.
OUTPUT_DATA_CSV = os.environ.get("OUTPUT_DATA_CSV", os.path.join(RUN_DIR, "raw", "advanced_synthetic_battery_data.csv"))
FAILURE_LOG_CSV = os.environ.get("FAILURE_LOG_CSV", os.path.join(RUN_DIR, "failures", "simulation_failures.csv"))
PULSE_PLOT_PNG = os.environ.get("PULSE_PLOT_PNG", os.path.join(RUN_DIR, "plots", "pulse_discharge_plot.png"))

for _output_path in (OUTPUT_DATA_CSV, FAILURE_LOG_CSV, PULSE_PLOT_PNG):
    os.makedirs(os.path.dirname(_output_path), exist_ok=True)

# Select the mathematical model (SPM). Deliberately isothermal, not PyBaMM's
# lumped-thermal option: that submodel requires entropic-heat and cell
# geometry parameters that Prada2013 (LFP) never defines, and even for
# Chen2020 (NMC) the isothermal model's Arrhenius kinetics are not wired to
# "Ambient temperature [K]" outside the thermal submodel (verified: sweeping
# 0-35 C produced byte-identical voltage output). Rather than force the
# thermal submodel and guess at LFP-specific entropic-heat data borrowed
# from an unrelated chemistry -- which would be actively misleading, not
# just approximate -- temperature's dominant real effect (higher internal
# resistance when cold) is applied explicitly below via TEMP_RESISTANCE_COEFF.
model = pybamm.lithium_ion.SPM()

# Load default chemical parameters. NMC is pooled across three independent
# literature parameter sets (not just Chen2020) so the model can't memorize
# one fixed OCP curve as a chemistry signature -- see NMC_PARAMETER_SETS
# below. ORegan2022 was also considered but excluded: its positive electrode
# conductivity is a temperature-dependent function, not a scalar, so it's
# incompatible with the *= scaling used for capacity/resistance below.
param_lfp_base = pybamm.ParameterValues("Prada2013")
NMC_PARAMETER_SETS = ["Chen2020", "Mohtat2020", "OKane2022"]
param_nmc_bases = {name: pybamm.ParameterValues(name) for name in NMC_PARAMETER_SETS}
param_nmc_base = param_nmc_bases["Chen2020"]  # reference for capacity_multipliers below

# Pulse discharge (GITT) from the article, green/yellow identification interval:
# total discharged capacity is fixed at ΔQ = 0.6 Ah (half of the assumed
# minimum 18650 capacity of 1.2 Ah), independent of the actual cell size,
# each pulse lasts 0.5 h and is followed by a 1 h rest. The per-step current
# is I = 0.6 Ah / n_steps / 0.5 h, which reproduces the article's stated
# value of 80 mA for 15 steps, and gives 240/120/60/48 mA for 5/10/20/25
# steps respectively. All of these are <=0.2C for the smallest (1.2 Ah)
# reference cell, i.e. "generally small values" as required by the article
# so the terminal voltage stays inside the feasible green/yellow measurement
# zone ([max(Vmin)-0.2V, min(Vmax)+0.2V] = [2.3V, 3.8V] for LFP/NMC, Sec. 2)
# instead of overshooting into the orange/red (deep-discharge/overcharge)
# zones during a pulse.
# Nominal lower voltage limits (article Sec. 2): LFP 2.0 V, NMC 2.5 V.
TOTAL_PULSE_CAPACITY_AH = 0.6
PULSE_DURATION = "30 minutes"
PULSE_DURATION_HOURS = 0.5
REST_DURATION = "1 hour"
LOWER_VOLTAGE_CUTOFF = {
    "LFP": 2.0,
    "NMC": 2.5,
}
STEP_COUNTS = [5, 10, 15, 20, 25]


def pulse_current_ma(n_steps):
    """Per-pulse current [mA] so n_steps pulses discharge the fixed total
    of TOTAL_PULSE_CAPACITY_AH, per the article's GITT protocol."""
    return TOTAL_PULSE_CAPACITY_AH / n_steps / PULSE_DURATION_HOURS * 1000


def make_pulse_experiment(n_steps, v_min):
    """Build a GITT discharge experiment with n pulse/rest cycles."""
    pulse_current = f"{pulse_current_ma(n_steps):.4g} mA"
    return pybamm.Experiment(
        [
            (
                f"Discharge at {pulse_current} for {PULSE_DURATION} or until {v_min} V",
                f"Rest for {REST_DURATION}",
            )
        ]
        * n_steps
    )


# Collects every solve failure (which chemistry/step-count/size/SOH, and what
# exception was raised) so the sample-count deficit between the theoretical
# chemistries*sizes*variations target and what actually lands in the CSV can
# be diagnosed after the run instead of only being visible as a missing row.
FAILURE_LOG = []


def solve_with_cutoff(sim, initial_soc, chem, n_steps, context=""):
    """Solve an experiment; keep partial results if the voltage cut-off is hit.

    Any solve failure is logged (with its exception type) and skipped rather
    than left to propagate. Previously only pybamm.SolverError was caught, so
    any other exception type raised by sim.solve() (e.g. from an extreme
    scaled-parameter combination) would crash the entire generation run
    without a trace, silently losing every remaining (chemistry, n_steps)
    attempt for that variation.
    """
    try:
        sol = sim.solve(initial_soc=initial_soc)
    except pybamm.SolverError as err:
        sol = getattr(sim, "solution", None)
        recovered = sol is not None and len(sol.t) > 0
        print(f"  {chem} ({n_steps} steps){context}: solver stopped early [SolverError] "
              f"({'partial data kept' if recovered else 'no data'}) ({err})")
        FAILURE_LOG.append({
            "Chemistry": chem, "N_Steps": n_steps, "Context": context,
            "ExceptionType": "SolverError", "Message": str(err), "Recovered": recovered,
        })
        if not recovered:
            return None
    except Exception as err:
        print(f"  {chem} ({n_steps} steps){context}: solve failed [{type(err).__name__}] ({err})")
        FAILURE_LOG.append({
            "Chemistry": chem, "N_Steps": n_steps, "Context": context,
            "ExceptionType": type(err).__name__, "Message": str(err), "Recovered": False,
        })
        return None

    termination = getattr(sol, "termination", None)
    if termination and termination != "final time":
        print(f"  {chem} ({n_steps} steps): terminated early ({termination})")

    return sol


# Target cell capacities from the article (Sec. 3): "the electrode geometries
# are adjusted to different capacities for training purposes, i.e., 1.2 Ah,
# 2 Ah, and 3.5 Ah, covering the common capacity range of 18650 cells."
# LFP (Prada2013, base ~2.3 Ah) and NMC (base ~5.0 Ah -- confirmed identical
# across all three pooled NMC_PARAMETER_SETS) have very different base
# capacities, so a single shared thickness multiplier does not land both
# chemistries on the same target capacity. Instead, a per-chemistry
# multiplier is derived from each base parameter set's own declared
# "Nominal cell capacity [A.h]", so that both chemistries actually reach
# ~1.2/2.0/3.5 Ah at each size step, keeping capacity itself uninformative
# about chemistry (as intended by the article).
CAPACITY_TARGETS_AH = [1.2, 2.0, 3.5]


def capacity_multipliers_for(base_params, targets_ah):
    """Per-chemistry thickness multipliers that scale a base parameter set's
    own nominal capacity onto each of the target capacities."""
    base_capacity_ah = base_params["Nominal cell capacity [A.h]"]
    return [target_ah / base_capacity_ah for target_ah in targets_ah]


capacity_multipliers = {
    "LFP": capacity_multipliers_for(param_lfp_base, CAPACITY_TARGETS_AH),
    "NMC": capacity_multipliers_for(param_nmc_base, CAPACITY_TARGETS_AH),
}

# How many random variations to run per battery size. The article (Sec. 3)
# generates 250 individual OCV samples per chemistry for each tested
# capacity/step configuration. With 2 chemistries (LFP, NMC) and 3 capacity
# sizes here, variations_per_size = 83 gives 3*83 = 249 samples per chemistry,
# matching that target (~10 min runtime measured for the full sweep).
variations_per_size = 83

all_data = []

print("Starting advanced simulations (Pulse Discharge, Random SOC/SOH)...")

for size_idx, target_ah in enumerate(CAPACITY_TARGETS_AH):
    for i in range(variations_per_size):
        # SOC is randomized over [SOC_RANGE_MIN, SOC_RANGE_MAX], 50-100% by
        # default, matching the article's own discharge-case sampling (Sec.
        # 3: "uniformly distributed random values between 50% and 100% SOC
        # for discharge"). run_soc_sweep.py overrides this range per run to
        # test how much high-voltage data a cell's starting SOC provides.
        # A previously fixed SOC (first 1.0, then 0.6) made every battery's pulse test
        # start from the identical point on the OCV curve, which both (a)
        # under-represented real-world variance and (b) is required now
        # that features are keyed on absolute voltage bins instead of
        # absolute SOC bins -- a fixed starting SOC would make each
        # chemistry only ever populate a narrow, unrealistically
        # consistent slice of the voltage range.
        # SOH is randomized over 50-85%: Stena is a recycling facility, so
        # the training population should reflect the degraded/end-of-life
        # cells it will actually receive, not a healthier range that would
        # never be seen at inference time (train/serve skew). Widened down
        # from the previous 75-85% band per explicit domain guidance.
        soc = random.uniform(0.5, 1.0)
        soh = random.uniform(0.50, 0.85)

        # Ambient temperature at time of test, independent of SOH. Real
        # degraded cells at the same nominal SOH don't all show the same
        # resistance: different aging pathways (SEI growth vs. lithium
        # plating vs. particle isolation vs. contact-resistance loss) and
        # different ambient conditions in storage/transport produce
        # different IR-drop behavior for the "same" capacity-based SOH.
        # Applied explicitly via TEMP_RESISTANCE_COEFF below rather than
        # PyBaMM's thermal submodel -- see the model-selection comment above.
        ambient_c = random.uniform(0, 35)
        TEMP_RESISTANCE_COEFF = 0.02  # ~2%/C conductivity change per degree
                                       # from a 25 C reference; a conservative,
                                       # commonly-cited order of magnitude for
                                       # Li-ion internal resistance vs. temperature
        temp_conductivity_multiplier = 1.0 + TEMP_RESISTANCE_COEFF * (ambient_c - 25.0)

        # Globally unique id for this (size, variation) draw. SOH is stored
        # rounded to 3 decimals, and with 83 random draws per size from only
        # ~101 possible rounded values, collisions are near-certain (birthday
        # paradox) -- two *different* variations can round to the identical
        # SOH, and since Initial_SOC and Size_Multiplier are otherwise
        # constant per size, that made them indistinguishable to every
        # downstream (Chemistry, Size_Multiplier, SOH, Initial_SOC) groupby,
        # silently merging independent simulation runs into one battery
        # group. Variation_ID is unaffected by any rounding and guarantees
        # each (size, i) draw stays its own identity everywhere downstream.
        variation_id = size_idx * variations_per_size + i

        # Pool NMC across independent literature parameter sets so the model
        # can't memorize one fixed OCP curve as a chemistry signature.
        nmc_set_name = random.choice(NMC_PARAMETER_SETS)

        print(f"\n--- Target size: {target_ah} Ah | Variation {i+1}/{variations_per_size} | "
              f"SOC: {soc:.2f} | SOH: {soh:.2f} | Ambient: {ambient_c:.1f}C | NMC set: {nmc_set_name} ---")

        # Create clean parameter copies
        param_lfp = param_lfp_base.copy()
        param_nmc = param_nmc_bases[nmc_set_name].copy()

        chemistries = {
            "LFP": param_lfp,
            "NMC": param_nmc,
        }

        # Apply per-chemistry Capacity Scaling (Electrode thickness) and Aging/SOH
        resistance_factors = {}
        for chem, param in chemistries.items():
            mult = capacity_multipliers[chem][size_idx]
            param["Negative electrode thickness [m]"] *= mult
            param["Positive electrode thickness [m]"] *= mult

            # 1. Capacity loss (LAM): reduce max lithium concentration
            param["Maximum concentration in negative electrode [mol.m-3]"] *= soh
            param["Maximum concentration in positive electrode [mol.m-3]"] *= soh

            # 2. Resistance growth: decoupled from SOH (not the same scalar),
            # so cells at the same nominal SOH don't all show identical
            # IR-drop -- see the ambient_c comment above. resistance_noise
            # gives each chemistry's resistance growth its own independent
            # draw (aging-pathway variance); temp_conductivity_multiplier
            # applies the shared ambient-temperature effect; the result is
            # clamped so conductivity never exceeds the un-aged baseline
            # (>1.0) or collapses to a solver-destabilizing near-zero value.
            resistance_noise = random.uniform(0.8, 1.2)
            resistance_factor = min(1.0, max(0.3, soh * resistance_noise * temp_conductivity_multiplier))
            resistance_factors[chem] = resistance_factor
            param["Negative electrode conductivity [S.m-1]"] *= resistance_factor
            param["Positive electrode conductivity [S.m-1]"] *= resistance_factor

        for n_steps in STEP_COUNTS:
            for chem, param in chemistries.items():
                v_min = LOWER_VOLTAGE_CUTOFF[chem]
                pulse_experiment = make_pulse_experiment(n_steps, v_min)
                sim = pybamm.Simulation(model, parameter_values=param, experiment=pulse_experiment)

                pulse_ma = pulse_current_ma(n_steps)
                print(f"Solving {chem} ({n_steps} steps, I={pulse_ma:.4g} mA, Vmin={v_min} V)...")
                context = f" [target={target_ah}Ah, variation={i+1}/{variations_per_size}, SOH={soh:.3f}]"
                sol = solve_with_cutoff(sim, soc, chem, n_steps, context=context)
                if sol is None:
                    print(f"  Skipping {chem} ({n_steps} steps): no solution data.")
                    continue

                raw_voltage = sol["Terminal voltage [V]"].entries
                noise = np.random.normal(0, 0.001, len(raw_voltage))
                voltage_with_noise = raw_voltage + noise

                df = pd.DataFrame({
                    "Time [s]": sol["Time [s]"].entries,
                    "Voltage [V]": voltage_with_noise,
                    "Capacity [A.h]": sol["Discharge capacity [A.h]"].entries,
                    "Chemistry": chem,
                    "Target_Capacity_Ah": target_ah,
                    "Size_Multiplier": capacity_multipliers[chem][size_idx],
                    "SOH": round(soh, 3),
                    "Initial_SOC": round(soc, 3),
                    "Ambient_Temperature_C": round(ambient_c, 2),
                    "Resistance_Factor": round(resistance_factors[chem], 4),
                    "Base_Parameter_Set": nmc_set_name if chem == "NMC" else "Prada2013",
                    "N_Steps": n_steps,
                    "V_min [V]": v_min,
                    "Variation_ID": variation_id,
                })
                all_data.append(df)

# Combine and save
training_data = pd.concat(all_data)
output_file = OUTPUT_DATA_CSV
training_data.to_csv(output_file, index=False)
print(f"\nDone! Data with realistic pulses, SOC, and aging saved to '{output_file}'")

# --- FAILURE SUMMARY ---
# Surfaces exactly how much of the theoretical chemistries*sizes*variations
# target was lost to solve failures, and why, instead of that deficit only
# showing up later as an unexplained gap in the feature-engineered dataset.
total_attempts = len(CAPACITY_TARGETS_AH) * variations_per_size * len(STEP_COUNTS) * len(LOWER_VOLTAGE_CUTOFF)
print(f"\n{len(FAILURE_LOG)} of {total_attempts} solve attempts failed.")
if FAILURE_LOG:
    failures_df = pd.DataFrame(FAILURE_LOG)
    print(failures_df["ExceptionType"].value_counts().to_string())
    failure_log_file = FAILURE_LOG_CSV
    failures_df.to_csv(failure_log_file, index=False)
    print(f"Full failure log saved to '{failure_log_file}'")

# --- PLOTTING ---
# Plot one LFP and one NMC sample to visualize the pulse discharge
print("Generating pulse discharge plot...")

# all_data[0] is the first LFP simulation
# all_data[1] is the first NMC simulation
lfp_sample = all_data[0]
nmc_sample = all_data[1]

plt.figure(figsize=(12, 6))
plt.plot(lfp_sample['Time [s]'] / 3600, lfp_sample['Voltage [V]'], label='LFP (Sample)', color='#1f77b4')
plt.plot(nmc_sample['Time [s]'] / 3600, nmc_sample['Voltage [V]'], label='NMC (Sample)', color='#ff7f0e')

plt.title('Simulated Pulse Discharge Profiles (GITT)')
plt.xlabel('Time [Hours]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid(True)

plot_file = PULSE_PLOT_PNG
plt.savefig(plot_file, dpi=150)
print(f"Plot saved to '{plot_file}'")

plt.show()
