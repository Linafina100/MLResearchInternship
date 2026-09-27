"""
Experiment 27: real-to-sim Initial_SOC sweep test set from the SNL
(Sandia National Labs) 18650 cylindrical cell dataset -- a 4th independent
real dataset alongside experiment 26's exp07/EMPA/CALCE, using the same
truncate_at_soc() technique (experiments/26_soc_sweep_three_datasets/soc_truncation.py).

DATASET: DownloadedData/SNL data/SNL LFP/ (21 cells) and SNL NMC/
(22 cells), each with a `*_timeseries.csv` (raw, per-sample) and a
`*_cycle_data.csv` (per-cycle summary, unused here). Filenames encode the
test condition, e.g. `SNL_18650_LFP_25C_0-100_0.5-1C_a_timeseries.csv` =
LFP chemistry, 25C, cycled over the full 0-100% SOC window, 0.5C
charge/1C discharge, replicate cell "a". Only `0-100` (full-range) files
are usable here -- the `20-80`/`40-60` files are partial-SOC-window aging
protocols that never traverse below their window floor, so they can never
supply a low-Initial_SOC sample. 43 full-range cells total (21 LFP + 22
NMC) -- a larger and more chemistry-balanced physical-cell count than any
other real dataset in this project (EMPA: 199 cells but coin-cell scale;
exp07: 4 cells, 1 LFP vs. 3 NMC; CALCE: 4 cells).

Unlike every other real-data build script in this project, SNL's
`Discharge_Capacity (Ah)` column is used DIRECTLY: verified by inspection
that it already resets to ~0 at the start of each cycle's discharge phase
and accumulates monotonically through it (each cycle here is
rest->charge->rest->discharge) -- no manual current-integration needed
(contrast experiment 07's LFP parser, which has to integrate current
itself).

SOH filtering (BOL from the first ~10 cycles, keep SOH in [0.8, 1.0]) is
applied here, unlike exp07 (never SOH-filtered in any prior experiment)
but matching EMPA/CALCE -- necessary both for consistency with the
synthetic training distribution and for tractability: these cells run
into the thousands of cycles each (one checked: 3546 cycles, fading from
1.032 Ah to ~0 Ah), so unfiltered this dataset would dwarf every other
real dataset in row count.

No RESCALE_TARGET_AH: SNL's native capacities (~1.0-1.1 Ah LFP, ~2.9 Ah
NMC) are already close to the synthetic training targets (1.2/2.0/3.5 Ah),
the same situation as experiment 07's cylindrical/pouch cells -- unlike
EMPA's mAh-scale coin cells, which need rescaling.

CAPACITY-BASED RESAMPLING (`resample_on_capacity()`, copied from EMPA's
build script) IS needed, though, for a reason specific to this dataset:
SNL's raw timeseries isn't logged at a fixed time interval -- higher
C-rate/higher-current cycles log far less densely in time (confirmed
directly: cycle 5 of one LFP cell had only 29 raw discharge samples total,
~120s apart, vs. ~700 samples at ~10s apart for a lower-rate early cycle).
LFP's end-of-discharge voltage collapse is characteristically almost a
step function in capacity space; at coarse, uneven raw sampling this cliff
can land inside just 1-2 raw samples, each covering several 0.1V bins at
once. Since feature_engineering.py bins point-to-point dV/dQ by the
sample's *ending* voltage, one such raw sample dumps one huge dV/dQ value
into whichever bin its endpoint happens to land in while the bins it
jumped clean over get zero real coverage from that trace -- confirmed
directly: real LFP's dV/dQ in the low-voltage bins was landing close to
real NMC's magnitude there (both saturating near -10 to -11), destroying
the classifier's signal (uniform LFP misclassification as NMC). Resampling
onto an evenly-spaced capacity grid (interpolated, following EMPA's
precedent) spreads each transition proportionally across the bins it
actually spans, fixing this.

No resimulation was needed for this dataset: it is, like EMPA/exp07/CALCE,
continuously-cycled lab data (never rested-then-resumed), so the existing
continuous-discharge-truncated synthetic training set
(data/26_soh_range_continuous_discharge_truncated_v1/) applies directly --
see experiments/26_soc_sweep_three_datasets/RESULTS.md and the
rested-vs-continuous-discharge project memory for why this distinction
matters.

Usage: python3 experiments/27_snl_soc_sweep/build_real_snl_soc_sweep.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SNL_DIR = os.path.join(PROJECT_DIR, "DownloadedData", "SNL data")
EXP26_DIR = os.path.join(PROJECT_DIR, "experiments", "26_soc_sweep_three_datasets")
sys.path.insert(0, EXP26_DIR)

from soc_truncation import truncate_at_soc

SOH_MIN = float(os.environ.get("SOH_MIN", 0.8))
SOH_MAX = float(os.environ.get("SOH_MAX", 1.0))
CURRENT_EPS_A = 5e-3
MIN_SEGMENT_POINTS = 15
BOL_WINDOW_CYCLES = 10
MAX_SEGMENT_DURATION_HOURS = 24.0
MIN_VOLTAGE_SPAN_V = 0.3
MAX_CYCLES_PER_CELL = int(os.environ.get("MAX_CYCLES_PER_CELL", 20))
N_RESAMPLE_POINTS = 80

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

CHEMISTRY_FOLDERS = {"SNL LFP": "LFP", "SNL NMC": "NMC"}
USECOLS = ["Test_Time (s)", "Cycle_Index", "Current (A)", "Voltage (V)",
           "Discharge_Capacity (Ah)", "Environment_Temperature (C)"]

RUN_LABEL = os.environ.get("RUN_LABEL", "27_real_snl_soc_sweep")
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

    early = [(n, seg) for n, seg in segments if n <= BOL_WINDOW_CYCLES]
    if not early:
        early = segments[:BOL_WINDOW_CYCLES]
    # MEDIAN, not max: SNL's cycle_data has a periodic (~every 500 cycles,
    # confirmed in all 43 files) doubled-capacity artifact -- a recurring
    # reference-performance-test cycle logged under a single Cycle_Index at
    # exactly ~2x the normal discharge capacity. This artifact always falls
    # within the first ~10 cycles too, so max() (which EMPA/CALCE use
    # deliberately to catch a real formation-capacity peak) would lock BOL
    # onto this artifact 100% of the time here. Median is robust to the
    # single outlier in the window; it also means these artifact cycles
    # naturally fail the SOH<=SOH_MAX filter later (their capacity/BOL is
    # ~2.0), so no separate special-casing is needed beyond this one line.
    bol_capacity = np.median([seg["Discharge_Capacity (Ah)"].iloc[-1] for _, seg in early])
    if bol_capacity <= 0:
        return None, f"{cell_id}: could not establish BOL capacity"

    qualifying = []
    for cycle_number, seg in segments:
        t_s = (seg["Test_Time (s)"].values - seg["Test_Time (s)"].iloc[0]).astype(np.float64)
        cap_ah = seg["Discharge_Capacity (Ah)"].values.astype(np.float64)
        # native column resets near 0 at each cycle's discharge start (verified by
        # inspection) but isn't pinned exactly -- anchor it the same way
        # segment_capacity_ah() does elsewhere in this project.
        cap_ah = cap_ah - cap_ah[0]
        capacity = cap_ah[-1]
        if capacity <= 0:
            continue
        soh = capacity / bol_capacity
        if not (SOH_MIN <= soh <= SOH_MAX):
            continue
        qualifying.append((cycle_number, seg, t_s, cap_ah, soh))

    # These cells run into the thousands of cycles, nearly all within
    # SOH 0.8-1.0 for most of their life (slow fade) -- unlike EMPA/CALCE,
    # where the SOH filter alone keeps cell counts small. Subsample to a
    # bounded number of cycles per cell, evenly spaced across the
    # qualifying cycles' *lifetime* (not just the earliest ones), so each
    # cell still contributes samples spanning its whole SOH 0.8-1.0 window
    # rather than clustering near BOL.
    if len(qualifying) > MAX_CYCLES_PER_CELL:
        idx = np.linspace(0, len(qualifying) - 1, MAX_CYCLES_PER_CELL).round().astype(int)
        qualifying = [qualifying[i] for i in sorted(set(idx))]

    rows = []
    n_cycles_matched = len(qualifying)
    n_variants = 0
    variant_counts_by_soc = {s: 0 for s in SOC_START_POINTS}
    for cycle_number, seg, t_s, cap_ah, soh in qualifying:
        voltage = seg["Voltage (V)"].values.astype(np.float64)
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

    status = (f"{cell_id} ({chemistry}): BOL={bol_capacity:.3f} Ah, "
              f"{len(segments)} discharge cycles scanned, {n_cycles_matched} cycles at "
              f"SOH in [{SOH_MIN},{SOH_MAX}], {n_variants} SOC-sweep variants kept {variant_counts_by_soc}")
    if not rows:
        return None, status
    return pd.concat(rows, ignore_index=True), status


def main():
    files = []
    for folder, chem in CHEMISTRY_FOLDERS.items():
        for fpath in sorted(glob.glob(os.path.join(SNL_DIR, folder, "*_0-100_*_timeseries.csv"))):
            files.append((fpath, chem))
    print(f"Found {len(files)} full-range ('0-100') SNL cell files")
    print(f"Target: SOH in [{SOH_MIN}, {SOH_MAX}]")
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
