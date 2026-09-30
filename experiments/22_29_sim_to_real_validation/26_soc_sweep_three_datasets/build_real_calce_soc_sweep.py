"""
Experiment 26: real-to-sim Initial_SOC sweep test set from the CALCE
dataset (4 real cells: 2 LFP + 2 NMC, single long discharge traces each --
see experiments/22_29_sim_to_real_validation/25_calce_sim_to_real/RESULTS.md for the column-collision
parser bug fixed there, reused unchanged here).

CALCE has no natural "cycle" grouping the way EMPA/exp07 do -- each of
the 4 files IS one single discharge trace, not a multi-cycle aging
series. So this script loops over the 4 files directly (not
discharge_segments()) and applies the same truncate_at_soc() technique
(experiments/22_29_sim_to_real_validation/23_empa_soc_sweep/build_real_empa_soc_sweep_dataset.py) to
each file's own capacity axis at each Initial_SOC point -- conceptually
valid (same formula: keep the portion after (1-s) of that trace's own
realized capacity has been delivered) even though there's only one trace
per cell to truncate, not many.

Does NOT additionally filter by the per-file SOH already computed at
parse time (capacity/BOL, clipped to [0.5, 1.0]) -- n=4 is already the
statistical ceiling for this dataset (see experiment 25's RESULTS.md);
filtering further risks zero usable samples. SOH is carried through as
metadata only.

CAVEAT (carried from experiment 25, applies identically here): n=4 real
cells means any accuracy number from this dataset is a spot-check, not
evidence of generalization -- a single flipped prediction is a 25-point
swing in raw accuracy.

Usage: python3 experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/build_real_calce_soc_sweep.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
EXP25_DIR = os.path.join(PROJECT_DIR, "experiments", "22_29_sim_to_real_validation", "25_calce_sim_to_real")
sys.path.insert(0, EXP25_DIR)
sys.path.insert(0, SCRIPT_DIR)

from build_calce_dataset import parse_calce_discharge_file
from soc_truncation import truncate_at_soc

CALCE_DIR = os.path.join(PROJECT_DIR, "data", "calce_experiment_data")
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

CALCE_CONFIGS = [
    {"pattern": "*A1-*", "chem": "LFP", "nominal_ah": 1.1},
    {"pattern": "*SP20*", "chem": "NMC", "nominal_ah": 2.0},
]

RUN_LABEL = os.environ.get("RUN_LABEL", "26_real_calce_soc_sweep")
OUT_RAW_CSV = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "raw", "real_calce_soc_sweep_raw.csv")

# truncate_at_soc() moved to soc_truncation.py (was identical across
# three build scripts; see that file's docstring).


def main():
    variation_counter = 1
    rows = []
    variant_counts_by_soc = {s: 0 for s in SOC_START_POINTS}

    for config in CALCE_CONFIGS:
        files = sorted(glob.glob(os.path.join(CALCE_DIR, config["pattern"])))
        files = [f for f in files if f.lower().endswith((".xlsx", ".xls", ".csv"))]
        for fpath in files:
            full_trace = parse_calce_discharge_file(
                file_path=fpath, chemistry=config["chem"],
                target_capacity_ah=config["nominal_ah"], variation_id=variation_counter,
            )
            variation_counter += 1

            t_s = full_trace['Time [s]'].values
            voltage = full_trace['Voltage [V]'].values
            cap_ah = full_trace['Capacity [A.h]'].values
            soh = full_trace['SOH'].iloc[0]
            source_file = os.path.basename(fpath)

            n_variants = 0
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
                    "Chemistry": config["chem"],
                    "Variation_ID": f"{source_file}_soc_{s}",
                    "Target_Capacity_Ah": config["nominal_ah"],
                    "Size_Multiplier": 1.0,
                    "SOH": soh,
                    "Initial_SOC": s,
                    "Ambient_Temperature_C": 25.0,
                    "Source_File": source_file,
                }))
            print(f"{source_file} ({config['chem']}, SOH={soh:.3f}): {n_variants}/{len(SOC_START_POINTS)} SOC variants kept")

    if not rows:
        print("\nNo variants kept anywhere.")
        return
    combined = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_RAW_CSV), exist_ok=True)
    combined.to_csv(OUT_RAW_CSV, index=False)

    n_variations = combined["Variation_ID"].nunique()
    print(f"\n{'=' * 70}")
    print(f"-> {OUT_RAW_CSV}")
    print(f"   {n_variations} real SOC-sweep variants (from 4 physical cells), {len(combined)} rows")
    print(f"   variants per SOC point: {variant_counts_by_soc}")
    print(combined.groupby(["Chemistry", "Initial_SOC"])["Variation_ID"].nunique())


if __name__ == "__main__":
    main()
