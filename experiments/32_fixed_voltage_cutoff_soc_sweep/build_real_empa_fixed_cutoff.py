"""
Experiment 32: real-to-sim Initial_SOC sweep test set from the EMPA
RO-Crate dataset, identical to
experiments/22_29_sim_to_real_validation/26_soc_sweep_three_datasets/build_real_empa_soc_sweep_soh_range.py
except for ONE change: BOL capacity and each cycle's own capacity are
both established by trimming to the fixed universal cutoff voltage (2.5V,
see this experiment's RESULTS.md) via
soc_truncation.trim_to_cutoff_voltage() BEFORE truncate_at_soc() is
called, instead of using each cycle's own raw final point as "0% SOC".

This cutoff voltage was chosen BECAUSE it's EMPA's own empirically
observed real stopping point (median per-cycle minimum voltage: 2.4998V
for both chemistries, confirmed directly from the raw data), so this
script's own numbers should barely move versus experiment 26's -- the
point of rebuilding it here is to use the exact same trimming mechanism
as exp07/SNL, and incidentally to clip off any trailing rest-phase
samples past the real cutoff (see experiment 22's RESULTS.md for that
known artifact) as a side effect of trimming exactly at the crossing.

Everything else (get_chemistry, discharge_segments, segment_capacity_ah,
resample_on_capacity, the rest-phase/duration/voltage-span sanity filters,
RESCALE_TARGET_AH, SOH range 0.8-1.0) is unchanged from experiment 26.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/build_real_empa_fixed_cutoff.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ROCRATE_DIR = os.path.join(PROJECT_DIR, "data", "Dataset-rocrate")
EXP26_DIR = os.path.join(PROJECT_DIR, "experiments", "22_29_sim_to_real_validation", "26_soc_sweep_three_datasets")
sys.path.insert(0, EXP26_DIR)

from soc_truncation import truncate_at_soc, trim_to_cutoff_voltage

CUTOFF_VOLTAGE = float(os.environ.get("CUTOFF_VOLTAGE", 2.5))
SOH_MIN = float(os.environ.get("SOH_MIN", 0.8))
SOH_MAX = float(os.environ.get("SOH_MAX", 1.0))
C_RATE_MIN = float(os.environ.get("C_RATE_MIN", 0.0))
C_RATE_MAX = float(os.environ.get("C_RATE_MAX", 999.0))
CURRENT_EPS_A = 5e-7
MIN_SEGMENT_POINTS = 15
BOL_WINDOW_CYCLES = 10
MAX_SEGMENT_DURATION_HOURS = 24.0
MIN_VOLTAGE_SPAN_V = 0.3
RESCALE_TARGET_AH = 2.0
N_RESAMPLE_POINTS = 80

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

ACTIVE_MATERIAL_TO_CHEMISTRY = {
    "LithiumIronPhosphateOxide": "LFP",
    "LithiumNickelCobaltManganeseOxide": "NMC",
}

RUN_LABEL = os.environ.get("RUN_LABEL", "32_real_empa_fixed_cutoff")
OUT_RAW_CSV = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "raw", "real_empa_soc_sweep_raw.csv")


def get_chemistry(cell_dir, cell_id):
    meta_path = os.path.join(cell_dir, f"{cell_id}.metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    try:
        pos = meta["@graph"][0]["hasTestObject"]["hasPositiveElectrode"]
        active_material = pos["hasCoating"]["hasActiveMaterial"]["rdfs:comment"]
    except (KeyError, IndexError):
        return None
    return ACTIVE_MATERIAL_TO_CHEMISTRY.get(active_material)


def discharge_segments(df):
    for cycle_number, cycle_df in df.groupby("cycle_dimensionless", sort=True):
        cycle_df = cycle_df.sort_values("test_time_millisecond", kind="stable")
        seg = cycle_df[cycle_df["current_ampere"] < -CURRENT_EPS_A]
        if len(seg) < MIN_SEGMENT_POINTS:
            continue
        seg = seg.reset_index(drop=True)

        duration_hours = (seg["test_time_millisecond"].iloc[-1] - seg["test_time_millisecond"].iloc[0]) / 3_600_000.0
        voltage_span = seg["voltage_volt"].max() - seg["voltage_volt"].min()
        if duration_hours > MAX_SEGMENT_DURATION_HOURS or voltage_span < MIN_VOLTAGE_SPAN_V:
            continue

        yield int(cycle_number), seg


def segment_capacity_ah(seg):
    t_hours = (seg["test_time_millisecond"] - seg["test_time_millisecond"].iloc[0]) / 3_600_000.0
    i_abs = seg["current_ampere"].abs().values
    cap = np.concatenate([[0.0], np.cumsum(
        np.diff(t_hours.values) * (i_abs[:-1] + i_abs[1:]) / 2.0
    )])
    return t_hours.values * 3600.0, cap


def resample_on_capacity(t_s, voltage, cap_ah, n_points=N_RESAMPLE_POINTS):
    cap_grid = np.linspace(cap_ah[0], cap_ah[-1], n_points)
    t_resampled = np.interp(cap_grid, cap_ah, t_s)
    v_resampled = np.interp(cap_grid, cap_ah, voltage)
    return t_resampled, v_resampled, cap_grid


def process_cell(cell_dir):
    cell_id = os.path.basename(cell_dir)
    chemistry = get_chemistry(cell_dir, cell_id)
    if chemistry is None:
        return None, f"{cell_id}: unrecognized/missing chemistry, skipped"

    parquet_path = os.path.join(cell_dir, f"{cell_id}.bdf.parquet")
    cols = ["test_time_millisecond", "current_ampere", "voltage_volt",
            "cycle_dimensionless", "ambient_temperature_celsius"]
    df = pd.read_parquet(parquet_path, columns=cols)
    for c in ("test_time_millisecond", "current_ampere", "voltage_volt", "ambient_temperature_celsius"):
        df[c] = df[c].astype("float32")

    segments = list(discharge_segments(df))
    if not segments:
        return None, f"{cell_id}: no valid discharge segments"

    early = [(n, seg) for n, seg in segments if n < BOL_WINDOW_CYCLES]
    if not early:
        early = segments[:BOL_WINDOW_CYCLES]
    bol_capacity = 0.0
    for _, seg in early:
        t_s, cap_ah = segment_capacity_ah(seg)
        voltage = seg["voltage_volt"].values.astype(np.float64)
        trimmed = trim_to_cutoff_voltage(t_s, voltage, cap_ah, CUTOFF_VOLTAGE)
        if trimmed is None:
            continue
        _, _, cap_trim = trimmed
        bol_capacity = max(bol_capacity, cap_trim[-1])
    if bol_capacity <= 0:
        return None, f"{cell_id}: could not establish BOL capacity at {CUTOFF_VOLTAGE}V cutoff"

    rows = []
    n_cycles_matched = 0
    n_variants = 0
    n_never_reaches_cutoff = 0
    variant_counts_by_soc = {s: 0 for s in SOC_START_POINTS}
    for cycle_number, seg in segments:
        t_s, cap_ah = segment_capacity_ah(seg)
        voltage = seg["voltage_volt"].values.astype(np.float64)

        trimmed = trim_to_cutoff_voltage(t_s, voltage, cap_ah, CUTOFF_VOLTAGE)
        if trimmed is None:
            n_never_reaches_cutoff += 1
            continue
        t_s, voltage, cap_ah = trimmed

        capacity = cap_ah[-1]
        if capacity <= 0:
            continue
        soh = capacity / bol_capacity
        mean_current_a = seg["current_ampere"].abs().mean()
        c_rate = mean_current_a / bol_capacity

        if not (SOH_MIN <= soh <= SOH_MAX):
            continue
        if not (C_RATE_MIN <= c_rate <= C_RATE_MAX):
            continue

        n_cycles_matched += 1
        ambient_t = float(seg["ambient_temperature_celsius"].mean())

        for s in SOC_START_POINTS:
            truncated = truncate_at_soc(t_s, voltage, cap_ah, s)
            if truncated is None:
                continue
            t_trunc, v_trunc, cap_trunc = truncated

            cap_trunc_rescaled = cap_trunc / bol_capacity * RESCALE_TARGET_AH
            t_rs, v_rs, cap_rs = resample_on_capacity(t_trunc, v_trunc, cap_trunc_rescaled)

            n_variants += 1
            variant_counts_by_soc[s] += 1
            rows.append(pd.DataFrame({
                "Time [s]": t_rs,
                "Voltage [V]": v_rs,
                "Capacity [A.h]": cap_rs,
                "Chemistry": chemistry,
                "Variation_ID": f"{cell_id}_cycle_{cycle_number}_soc_{s}",
                "Target_Capacity_Ah": RESCALE_TARGET_AH,
                "Real_BOL_Capacity_mAh": bol_capacity * 1000,
                "Size_Multiplier": 0.0,
                "SOH": round(float(soh), 4),
                "Initial_SOC": s,
                "Ambient_Temperature_C": ambient_t,
            }))

    status = (f"{cell_id} ({chemistry}): BOL={bol_capacity * 1000:.3f} mAh @ {CUTOFF_VOLTAGE}V, "
              f"{len(segments)} discharge cycles scanned ({n_never_reaches_cutoff} never reach cutoff), "
              f"{n_cycles_matched} cycles at SOH in [{SOH_MIN},{SOH_MAX}] & C-rate [{C_RATE_MIN},{C_RATE_MAX}], "
              f"{n_variants} SOC-sweep variants kept {variant_counts_by_soc}")
    if not rows:
        return None, status
    return pd.concat(rows, ignore_index=True), status


def main():
    cell_dirs = sorted(d for d in glob.glob(os.path.join(ROCRATE_DIR, "empa__ccid*")) if os.path.isdir(d))
    print(f"Found {len(cell_dirs)} cell directories in '{ROCRATE_DIR}'")
    print(f"Target: SOH in [{SOH_MIN}, {SOH_MAX}], C-rate [{C_RATE_MIN}, {C_RATE_MAX}], cutoff {CUTOFF_VOLTAGE}V")
    print(f"SOC_START_POINTS: {SOC_START_POINTS}\n")

    all_rows = []
    n_lfp_cells = n_nmc_cells = 0
    for i, cell_dir in enumerate(cell_dirs, 1):
        out, status = process_cell(cell_dir)
        print(f"[{i}/{len(cell_dirs)}] {status}")
        if out is not None:
            all_rows.append(out)
            if out["Chemistry"].iloc[0] == "LFP":
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
