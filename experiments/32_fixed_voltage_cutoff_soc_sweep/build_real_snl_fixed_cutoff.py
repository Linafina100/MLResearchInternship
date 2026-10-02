"""
Experiment 32: real-to-sim Initial_SOC sweep test set from the SNL
(Sandia National Labs) 18650 cylindrical cell dataset, identical to
experiments/22_29_sim_to_real_validation/27_snl_soc_sweep/build_real_snl_soc_sweep.py
(lead-in-artifact fix included unchanged -- see that script's docstring)
except for ONE further change: BOL capacity and each cycle's own capacity
are both established by trimming to the fixed universal cutoff voltage
(2.5V, see this experiment's RESULTS.md) via
soc_truncation.trim_to_cutoff_voltage() BEFORE truncate_at_soc() is
called, instead of using each cycle's own raw final point as "0% SOC".

UNLIKE exp07/EMPA (whose real cells already stop almost exactly at 2.5V
on their own), SNL's cells are rated lower and natively discharge well
past 2.5V, down to ~2.0V. Trimming at 2.5V here means genuinely discarding
the deepest ~0.5V of SNL's real data from this experiment's 0-100% window
-- including the region that earlier experiments (27/29) found carried
SNL's most LFP-diagnostic (and most sim-mismatched) signal. That's an
intentional, explicit tradeoff for this experiment (see RESULTS.md):
project-wide comparability against one fixed physical voltage, at the
cost of SNL's deepest region.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/build_real_snl_fixed_cutoff.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SNL_DIR = os.path.join(PROJECT_DIR, "DownloadedData", "SNL data")
EXP26_DIR = os.path.join(PROJECT_DIR, "experiments", "22_29_sim_to_real_validation", "26_soc_sweep_three_datasets")
sys.path.insert(0, EXP26_DIR)

from soc_truncation import truncate_at_soc, trim_to_cutoff_voltage

CUTOFF_VOLTAGE = float(os.environ.get("CUTOFF_VOLTAGE", 2.5))
SOH_MIN = float(os.environ.get("SOH_MIN", 0.8))
SOH_MAX = float(os.environ.get("SOH_MAX", 1.0))
CURRENT_EPS_A = 5e-3
MIN_SEGMENT_POINTS = 15
BOL_WINDOW_CYCLES = 10
MAX_SEGMENT_DURATION_HOURS = 24.0
MIN_VOLTAGE_SPAN_V = 0.3
# See experiments/22_29_sim_to_real_validation/27_snl_soc_sweep/build_real_snl_soc_sweep.py
# for why this exists -- unchanged here.
LEAD_IN_GAP_THRESHOLD_S = 1800.0
MAX_CYCLES_PER_CELL = int(os.environ.get("MAX_CYCLES_PER_CELL", 20))
N_RESAMPLE_POINTS = 80

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

CHEMISTRY_FOLDERS = {"SNL LFP": "LFP", "SNL NMC": "NMC"}
USECOLS = ["Test_Time (s)", "Cycle_Index", "Current (A)", "Voltage (V)",
           "Discharge_Capacity (Ah)", "Environment_Temperature (C)"]

RUN_LABEL = os.environ.get("RUN_LABEL", "32_real_snl_fixed_cutoff")
OUT_RAW_CSV = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "raw", "real_snl_soc_sweep_raw.csv")


def resample_on_capacity(t_s, voltage, cap_ah, n_points=N_RESAMPLE_POINTS):
    cap_grid = np.linspace(cap_ah[0], cap_ah[-1], n_points)
    t_resampled = np.interp(cap_grid, cap_ah, t_s)
    v_resampled = np.interp(cap_grid, cap_ah, voltage)
    return t_resampled, v_resampled, cap_grid


def discharge_segments(df):
    for cycle_number, cycle_df in df.groupby("Cycle_Index", sort=True):
        cycle_df = cycle_df.sort_values("Test_Time (s)", kind="stable")
        seg = cycle_df[cycle_df["Current (A)"] < -CURRENT_EPS_A]
        if len(seg) < MIN_SEGMENT_POINTS:
            continue
        seg = seg.reset_index(drop=True)

        # Drop a leftover leading (or trailing) fragment from a different
        # physical event (see experiment 27's build script). Split on any
        # gap that large and keep only the single largest contiguous run,
        # which is always the cycle's real, continuous discharge.
        gaps = np.diff(seg["Test_Time (s)"].values)
        if len(gaps) > 0 and gaps.max() > LEAD_IN_GAP_THRESHOLD_S:
            split_points = np.where(gaps > LEAD_IN_GAP_THRESHOLD_S)[0] + 1
            runs = np.split(np.arange(len(seg)), split_points)
            seg = seg.iloc[max(runs, key=len)].reset_index(drop=True)
            if len(seg) < MIN_SEGMENT_POINTS:
                continue

        duration_hours = (seg["Test_Time (s)"].iloc[-1] - seg["Test_Time (s)"].iloc[0]) / 3600.0
        voltage_span = seg["Voltage (V)"].max() - seg["Voltage (V)"].min()
        if duration_hours > MAX_SEGMENT_DURATION_HOURS or voltage_span < MIN_VOLTAGE_SPAN_V:
            continue

        yield int(cycle_number), seg


def process_cell(file_path, chemistry):
    cell_id = os.path.basename(file_path).replace("_timeseries.csv", "")
    df = pd.read_csv(file_path, usecols=USECOLS)
    for c in USECOLS[1:]:
        df[c] = df[c].astype("float32")

    segments = list(discharge_segments(df))
    if not segments:
        return None, f"{cell_id}: no valid discharge segments"

    def raw_cap_and_voltage(seg):
        t_s = (seg["Test_Time (s)"].values - seg["Test_Time (s)"].iloc[0]).astype(np.float64)
        cap_ah = seg["Discharge_Capacity (Ah)"].values.astype(np.float64)
        cap_ah = cap_ah - cap_ah[0]  # anchor, as in experiment 27
        voltage = seg["Voltage (V)"].values.astype(np.float64)
        return t_s, voltage, cap_ah

    early = [(n, seg) for n, seg in segments if n <= BOL_WINDOW_CYCLES]
    if not early:
        early = segments[:BOL_WINDOW_CYCLES]
    # MEDIAN, not max -- see experiment 27's build script for why (robust
    # to the periodic doubled-capacity reference-performance-test
    # artifact). Computed here on the CUTOFF-TRIMMED capacity.
    early_trimmed_caps = []
    for _, seg in early:
        t_s, voltage, cap_ah = raw_cap_and_voltage(seg)
        trimmed = trim_to_cutoff_voltage(t_s, voltage, cap_ah, CUTOFF_VOLTAGE)
        if trimmed is None:
            continue
        _, _, cap_trim = trimmed
        early_trimmed_caps.append(cap_trim[-1])
    if not early_trimmed_caps:
        return None, f"{cell_id}: could not establish BOL capacity at {CUTOFF_VOLTAGE}V cutoff"
    bol_capacity = np.median(early_trimmed_caps)
    if bol_capacity <= 0:
        return None, f"{cell_id}: could not establish BOL capacity"

    qualifying = []
    n_never_reaches_cutoff = 0
    for cycle_number, seg in segments:
        t_s, voltage, cap_ah = raw_cap_and_voltage(seg)
        trimmed = trim_to_cutoff_voltage(t_s, voltage, cap_ah, CUTOFF_VOLTAGE)
        if trimmed is None:
            n_never_reaches_cutoff += 1
            continue
        t_s, voltage, cap_ah = trimmed
        capacity = cap_ah[-1]
        if capacity <= 0:
            continue
        soh = capacity / bol_capacity
        if not (SOH_MIN <= soh <= SOH_MAX):
            continue
        qualifying.append((cycle_number, seg, t_s, voltage, cap_ah, soh))

    # Subsample to a bounded number of cycles per cell, evenly spaced
    # across the qualifying cycles' lifetime -- see experiment 27's build
    # script for why.
    if len(qualifying) > MAX_CYCLES_PER_CELL:
        idx = np.linspace(0, len(qualifying) - 1, MAX_CYCLES_PER_CELL).round().astype(int)
        qualifying = [qualifying[i] for i in sorted(set(idx))]

    rows = []
    n_cycles_matched = len(qualifying)
    n_variants = 0
    variant_counts_by_soc = {s: 0 for s in SOC_START_POINTS}
    for cycle_number, seg, t_s, voltage, cap_ah, soh in qualifying:
        ambient_t = float(seg["Environment_Temperature (C)"].mean())

        for s in SOC_START_POINTS:
            truncated = truncate_at_soc(t_s, voltage, cap_ah, s)
            if truncated is None:
                continue
            t_trunc, v_trunc, cap_trunc = truncated
            t_rs, v_rs, cap_rs = resample_on_capacity(t_trunc, v_trunc, cap_trunc)
            n_variants += 1
            variant_counts_by_soc[s] += 1
            rows.append(pd.DataFrame({
                "Time [s]": t_rs,
                "Voltage [V]": v_rs,
                "Capacity [A.h]": cap_rs,
                "Chemistry": chemistry,
                "Variation_ID": f"{cell_id}_cycle_{cycle_number}_soc_{s}",
                "Target_Capacity_Ah": bol_capacity,
                "Size_Multiplier": 0.0,
                "SOH": round(float(soh), 4),
                "Initial_SOC": s,
                "Ambient_Temperature_C": ambient_t,
            }))

    status = (f"{cell_id} ({chemistry}): BOL={bol_capacity:.3f} Ah @ {CUTOFF_VOLTAGE}V, "
              f"{len(segments)} discharge cycles scanned ({n_never_reaches_cutoff} never reach cutoff), "
              f"{n_cycles_matched} cycles at SOH in [{SOH_MIN},{SOH_MAX}], "
              f"{n_variants} SOC-sweep variants kept {variant_counts_by_soc}")
    if not rows:
        return None, status
    return pd.concat(rows, ignore_index=True), status


def main():
    files = []
    for folder, chem in CHEMISTRY_FOLDERS.items():
        for fpath in sorted(glob.glob(os.path.join(SNL_DIR, folder, "*_0-100_*_timeseries.csv"))):
            files.append((fpath, chem))
    print(f"Found {len(files)} full-range ('0-100') SNL cell files")
    print(f"Target: SOH in [{SOH_MIN}, {SOH_MAX}], cutoff {CUTOFF_VOLTAGE}V")
    print(f"SOC_START_POINTS: {SOC_START_POINTS}\n")

    all_rows = []
    n_lfp_cells = n_nmc_cells = 0
    for i, (fpath, chem) in enumerate(files, 1):
        out, status = process_cell(fpath, chem)
        print(f"[{i}/{len(files)}] {status}")
        if out is not None:
            all_rows.append(out)
            if chem == "LFP":
                n_lfp_cells += 1
            else:
                n_nmc_cells += 1

    if not all_rows:
        print("\nNo matching cycles found anywhere.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_RAW_CSV), exist_ok=True)
    combined.to_csv(OUT_RAW_CSV, index=False)

    n_variations = combined["Variation_ID"].nunique()
    print(f"\n{'=' * 70}")
    print(f"-> {OUT_RAW_CSV}")
    print(f"   {n_variations} real SOC-sweep variants kept "
          f"({n_lfp_cells} LFP cells, {n_nmc_cells} NMC cells contributing), "
          f"{len(combined)} rows")
    print(combined.groupby(["Chemistry", "Initial_SOC"])["Variation_ID"].nunique())


if __name__ == "__main__":
    main()
