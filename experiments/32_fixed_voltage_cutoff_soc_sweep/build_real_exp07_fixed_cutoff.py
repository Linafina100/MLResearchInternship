"""
Experiment 32: real-to-sim Initial_SOC sweep test set from experiment 07's
original real dataset, identical to
experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/build_real_exp07_soc_sweep.py
except for ONE change: each cycle is trimmed to end at the fixed universal
cutoff voltage (2.5V, see this experiment's RESULTS.md) via
soc_truncation.trim_to_cutoff_voltage() BEFORE truncate_at_soc() is
called, instead of using the cycle's own raw final point as "0% SOC".

In practice this barely changes exp07's own numbers -- its real cells
already stop almost exactly at 2.5V on their own (median per-cycle minimum
voltage: 2.496V LFP, 2.4999V NMC, confirmed directly from the raw data) --
but it's included for methodological consistency with EMPA and SNL, which
this experiment also rebuilds against the same fixed cutoff.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/build_real_exp07_fixed_cutoff.py
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXP07_DIR = os.path.join(PROJECT_DIR, "experiments", "07_real_lfp_nmc_test")
EXP26_DIR = os.path.join(PROJECT_DIR, "experiments", "22_29_sim_to_real_validation", "26_soc_sweep_three_datasets")
sys.path.insert(0, EXP07_DIR)
sys.path.insert(0, EXP26_DIR)

from parse_real_lfp import parse_lfp_discharge_files
from parse_real_nmc import parse_nmc_files
from soc_truncation import truncate_at_soc, trim_to_cutoff_voltage, MIN_RAW_POINTS_AFTER_TRUNCATION

CUTOFF_VOLTAGE = float(os.environ.get("CUTOFF_VOLTAGE", 2.5))
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

RUN_LABEL = os.environ.get("RUN_LABEL", "32_real_exp07_fixed_cutoff")
OUT_RAW_CSV = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "raw", "real_exp07_soc_sweep_raw.csv")


def sweep_one_chemistry(df, chemistry):
    rows = []
    n_cycles = n_variants = 0
    variant_counts_by_soc = {s: 0 for s in SOC_START_POINTS}
    skipped_degenerate = 0
    skipped_never_reaches_cutoff = 0

    for variation_id, cycle_df in df.groupby('Variation_ID'):
        cycle_df = cycle_df.sort_values('Time [s]', kind='stable').reset_index(drop=True)
        t_s = cycle_df['Time [s]'].values
        voltage = cycle_df['Voltage [V]'].values
        cap_ah = cycle_df['Capacity [A.h]'].values

        if len(cycle_df) < MIN_RAW_POINTS_AFTER_TRUNCATION or cap_ah[-1] <= 0:
            skipped_degenerate += 1
            continue

        trimmed = trim_to_cutoff_voltage(t_s, voltage, cap_ah, CUTOFF_VOLTAGE)
        if trimmed is None:
            skipped_never_reaches_cutoff += 1
            continue
        t_s, voltage, cap_ah = trimmed

        n_cycles += 1
        for s in SOC_START_POINTS:
            truncated = truncate_at_soc(t_s, voltage, cap_ah, s)
            if truncated is None:
                continue
            t_trunc, v_trunc, cap_trunc = truncated
            n_variants += 1
            variant_counts_by_soc[s] += 1
            rows.append(pd.DataFrame({
                "Time [s]": t_trunc,
                "Voltage [V]": v_trunc,
                "Capacity [A.h]": cap_trunc,
                "Chemistry": chemistry,
                "Variation_ID": f"{variation_id}_soc_{s}",
                "Target_Capacity_Ah": cap_ah[-1],
                "Size_Multiplier": 0.0,
                "SOH": 0.0,  # placeholder -- see experiment 26's docstring: no real per-cycle SOH exists for this dataset
                "Initial_SOC": s,
            }))

    print(f"  {chemistry}: {n_cycles} usable cycles ({skipped_degenerate} degenerate, "
          f"{skipped_never_reaches_cutoff} never reach {CUTOFF_VOLTAGE}V, skipped), "
          f"{n_variants} SOC-sweep variants kept {variant_counts_by_soc}")
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def main():
    print("Loading experiment 07's real LFP data...")
    lfp_df = parse_lfp_discharge_files()
    print("Loading experiment 07's real NMC data...")
    nmc_df = parse_nmc_files()

    print(f"\nSweeping Initial_SOC {SOC_START_POINTS} (fixed cutoff {CUTOFF_VOLTAGE}V)...")
    lfp_out = sweep_one_chemistry(lfp_df, "LFP")
    nmc_out = sweep_one_chemistry(nmc_df, "NMC")

    parts = [p for p in (lfp_out, nmc_out) if p is not None]
    if not parts:
        print("\nNo variants kept anywhere.")
        return
    combined = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_RAW_CSV), exist_ok=True)
    combined.to_csv(OUT_RAW_CSV, index=False)

    n_variations = combined["Variation_ID"].nunique()
    print(f"\n{'=' * 70}")
    print(f"-> {OUT_RAW_CSV}")
    print(f"   {n_variations} real SOC-sweep variants, {len(combined)} rows")
    print(combined.groupby(["Chemistry", "Initial_SOC"])["Variation_ID"].nunique())


if __name__ == "__main__":
    main()
