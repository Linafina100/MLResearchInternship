"""
Experiment 24: same EMPA RO-Crate parsing as experiment 22's
build_real_empa_dataset.py (data/Dataset-rocrate/, 199 coin cells, 167
NMC / 32 LFP), but selecting real cycles by a SOH RANGE (default
[0.8, 1.0]) and generating truncated versions for different Initial_SOC levels.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EMPA_DIR = os.path.join(PROJECT_DIR, "data", "Empa_dataset")

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

ACTIVE_MATERIAL_TO_CHEMISTRY = {
    "LithiumIronPhosphateOxide": "LFP",
    "LithiumNickelCobaltManganeseOxide": "NMC",
}

OUT_RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")


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
        _, cap = segment_capacity_ah(seg)
        bol_capacity = max(bol_capacity, cap[-1])
    if bol_capacity <= 0:
        return None, f"{cell_id}: could not establish BOL capacity"

    cell_cycles = []
    for cycle_number, seg in segments:
        t_s, cap_ah = segment_capacity_ah(seg)
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

        cell_cycles.append({
            "cycle_number": cycle_number,
            "seg": seg,
            "t_s": t_s,
            "cap_ah": cap_ah,
            "capacity": capacity,
            "bol_capacity": bol_capacity,
            "soh": soh,
            "chemistry": chemistry,
            "cell_id": cell_id
        })

    status = f"{cell_id} ({chemistry}): {len(cell_cycles)} matched cycles found"
    return cell_cycles, status


def main():
    cell_dirs = sorted(d for d in glob.glob(os.path.join(EMPA_DIR, "empa__ccid*")) if os.path.isdir(d))
    print(f"Found {len(cell_dirs)} cell directories in '{EMPA_DIR}'")
    print(f"Target: SOH in [{SOH_MIN}, {SOH_MAX}], C-rate [{C_RATE_MIN}, {C_RATE_MAX}]\n")

    all_cell_data = []
    for i, cell_dir in enumerate(cell_dirs, 1):
        cell_cycles, status = process_cell(cell_dir)
        print(f"[{i}/{len(cell_dirs)}] {status}")
        if cell_cycles:
            all_cell_data.extend(cell_cycles)

    if not all_cell_data:
        print("\nNo matching cycles found anywhere -- widen SOH_MIN/SOH_MAX or the C-rate window.")
        return

    soc_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    os.makedirs(OUT_RAW_DIR, exist_ok=True)

    # Bygg dataset för varje Initial_SOC-nivå genom att kapa kurvorna
    for target_soc in soc_levels:
        rows = []
        for item in all_cell_data:
            seg = item["seg"]
            t_s = item["t_s"]
            cap_ah = item["cap_ah"]
            bol_capacity = item["bol_capacity"]
            soh = item["soh"]
            chemistry = item["chemistry"]
            cell_id = item["cell_id"]
            cycle_number = item["cycle_number"]

            total_cap = cap_ah[-1]
            if total_cap <= 0:
                continue

            # Kapa bort början av urladdningen upp till (1.0 - target_soc)
            cutoff_capacity = total_cap * (1.0 - target_soc)
            mask = cap_ah >= cutoff_capacity
            if mask.sum() < MIN_SEGMENT_POINTS:
                continue

            seg_trunc = seg[mask].reset_index(drop=True)
            t_s_filtered = t_s[mask]
            t_s_trunc = t_s_filtered - t_s_filtered[0]
            cap_ah_filtered = cap_ah[mask]
            cap_ah_trunc = cap_ah_filtered - cutoff_capacity

            if len(cap_ah_trunc) < 2 or (cap_ah_trunc[-1] - cap_ah_trunc[0]) <= 0:
                continue

            cap_ah_rescaled = cap_ah_trunc / bol_capacity * RESCALE_TARGET_AH
            t_rs, v_rs, cap_rs = resample_on_capacity(t_s_trunc, seg_trunc["voltage_volt"].values, cap_ah_rescaled)

            rows.append(pd.DataFrame({
                "Time [s]": t_rs,
                "Voltage [V]": v_rs,
                "Capacity [A.h]": cap_rs,
                "Chemistry": chemistry,
                "Variation_ID": f"{cell_id}_cycle_{cycle_number}_soc_{int(target_soc*100)}",
                "Target_Capacity_Ah": RESCALE_TARGET_AH,
                "Real_BOL_Capacity_mAh": bol_capacity * 1000,
                "Size_Multiplier": 0.0,
                "SOH": round(float(soh), 4),
                "Initial_SOC": target_soc,
                "Ambient_Temperature_C": float(seg_trunc["ambient_temperature_celsius"].mean()),
            }))

        if rows:
            combined_soc = pd.concat(rows, ignore_index=True)
            out_path = os.path.join(OUT_RAW_DIR, f"real_empa_raw_soc_{int(target_soc*100)}.csv")
            combined_soc.to_csv(out_path, index=False)
            print(f"Skapade kapat dataset för SOC {int(target_soc*100)}%: {len(combined_soc)} rader ({combined_soc['Variation_ID'].nunique()} cykler)")

    # Spara även standardfilen för 100% som bakåtkompatibilitet
    default_100_path = os.path.join(OUT_RAW_DIR, "real_empa_raw.csv")
    if os.path.exists(os.path.join(OUT_RAW_DIR, "real_empa_raw_soc_100.csv")):
        import shutil
        shutil.copy(os.path.join(OUT_RAW_DIR, "real_empa_raw_soc_100.csv"), default_100_path)

    print(f"\n{'=' * 70}")
    print("Alla kapade SOC-dataset har genererats i mappen data/raw/!")


if __name__ == "__main__":
    main()