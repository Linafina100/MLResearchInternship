"""
Experiment 26: real-to-sim Initial_SOC sweep test set from experiment 07's
original real dataset (4 physical cells: 1 LFP + 3 NMC, 579 + 1782 raw
discharge cycles), using the same truncate_at_soc() technique
experiments/23_empa_soc_sweep/build_real_empa_soc_sweep_dataset.py
introduced for EMPA.

No SOC-sweep script exists for this dataset today. Reuses
parse_lfp_discharge_files() / parse_nmc_files()
(experiments/07_real_lfp_nmc_test/) UNCHANGED -- confirmed these already
produce a monotonically-increasing Capacity [A.h] per cycle, exactly what
truncate_at_soc() needs. No RESCALE_TARGET_AH/resampling step: unlike
EMPA's mAh-scale coin cells, these cells' native capacity is already
Ah-scale and close to the synthetic training targets (LFP ~5Ah, NMC
~2.8Ah measured directly), matching why experiments 16/17/20/21 never
needed to rescale this dataset either.

No SOH filtering is applied -- matching established precedent: every
prior experiment using this dataset (16/17/20/21) used ALL of its cycles
unfiltered by degradation state; its SOH column has always been a
constant placeholder (0.0), not a real per-cycle value the way EMPA/CALCE
compute one. This is a genuine inconsistency versus how the other two
real datasets in this sweep are selected -- stated plainly in RESULTS.md,
not silently patched over by inventing a new filter this dataset was
never subject to before.

IMPORTANT CAVEAT (see experiments/07_real_lfp_nmc_test/01_real_vs_real_device_confound/RESULTS.md):
this dataset has only 1 LFP cell vs. 3 NMC cells -- chemistry label is
entangled with device/lab identity. That finding was about splitting
THIS dataset's own data into train/test by cycle, letting a model
memorize a device fingerprint. This experiment only ever uses this
dataset as a held-out REAL TEST SET for a model trained purely on
synthetic data, which never sees this dataset's device fingerprint during
training -- the specific shortcut that finding describes is not available
here. The underlying statistical-power concern (only 4 independent
physical cells) still stands and must be stated in RESULTS.md.

Usage: python3 experiments/26_soc_sweep_three_datasets/build_real_exp07_soc_sweep.py
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXP07_DIR = os.path.join(PROJECT_DIR, "experiments", "07_real_lfp_nmc_test")
sys.path.insert(0, EXP07_DIR)

from parse_real_lfp import parse_lfp_discharge_files
from parse_real_nmc import parse_nmc_files

MIN_RAW_POINTS_AFTER_TRUNCATION = 5
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

RUN_LABEL = os.environ.get("RUN_LABEL", "26_real_exp07_soc_sweep")
OUT_RAW_CSV = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "raw", "real_exp07_soc_sweep_raw.csv")


def truncate_at_soc(t_s, voltage, cap_ah, soc_start):
    """Same technique as experiment 23's build script (see that file's
    docstring) -- keeps the portion of a full real discharge trace after
    (1 - soc_start) of THIS cycle's own realized capacity has already
    been delivered, resetting Time/Capacity to 0 at the new start."""
    total_capacity = cap_ah[-1]
    cap_threshold = (1.0 - soc_start) * total_capacity

    if cap_threshold <= 0:
        t_kept, v_kept, cap_kept = t_s, voltage, cap_ah
    else:
        t_at_thresh = np.interp(cap_threshold, cap_ah, t_s)
        v_at_thresh = np.interp(cap_threshold, cap_ah, voltage)
        mask = cap_ah > cap_threshold
        if mask.sum() < MIN_RAW_POINTS_AFTER_TRUNCATION - 1:
            return None
        t_kept = np.concatenate([[t_at_thresh], t_s[mask]])
        v_kept = np.concatenate([[v_at_thresh], voltage[mask]])
        cap_kept = np.concatenate([[cap_threshold], cap_ah[mask]])
        t_kept = t_kept - t_kept[0]
        cap_kept = cap_kept - cap_kept[0]

    if len(t_kept) < MIN_RAW_POINTS_AFTER_TRUNCATION:
        return None

    return t_kept, v_kept, cap_kept


def sweep_one_chemistry(df, chemistry):
    rows = []
    n_cycles = n_variants = 0
    variant_counts_by_soc = {s: 0 for s in SOC_START_POINTS}
    skipped_degenerate = 0

    for variation_id, cycle_df in df.groupby('Variation_ID'):
        cycle_df = cycle_df.sort_values('Time [s]', kind='stable').reset_index(drop=True)
        t_s = cycle_df['Time [s]'].values
        voltage = cycle_df['Voltage [V]'].values
        cap_ah = cycle_df['Capacity [A.h]'].values

        if len(cycle_df) < MIN_RAW_POINTS_AFTER_TRUNCATION or cap_ah[-1] <= 0:
            skipped_degenerate += 1
            continue

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
                "SOH": 0.0,  # placeholder -- see docstring: no real per-cycle SOH exists for this dataset
                "Initial_SOC": s,
            }))

    print(f"  {chemistry}: {n_cycles} usable cycles ({skipped_degenerate} degenerate/skipped), "
          f"{n_variants} SOC-sweep variants kept {variant_counts_by_soc}")
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def main():
    print("Loading experiment 07's real LFP data...")
    lfp_df = parse_lfp_discharge_files()
    print("Loading experiment 07's real NMC data...")
    nmc_df = parse_nmc_files()

    print(f"\nSweeping Initial_SOC {SOC_START_POINTS}...")
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
