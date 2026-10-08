"""
EMPA RO-Crate builder with SOC sweep truncation.
Extracts real cycles matching SOH_MIN <= SOH <= SOH_MAX, applies truncation across
SOC_START_POINTS without target-zone restrictions, and outputs real_empa_raw.csv.
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
MIN_RAW_POINTS_AFTER_TRUNCATION = 5

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

ACTIVE_MATERIAL_TO_CHEMISTRY = {
    "LithiumIronPhosphateOxide": "LFP",
    "LithiumNickelCobaltManganeseOxide": "NMC",
}

OUT_RAW_CSV = os.path.join(PROJECT_DIR, "data", "raw", "real_empa_raw.csv")


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


def truncate_at_soc(t_s, voltage, cap_ah, soc_start):
    """Keeps only the portion of a full real discharge trace occurring after
    (1 - soc_start) of this cycle's delivered capacity. Time and Capacity are
    reset to 0. No target-zone voltage boundary check is enforced."""
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

    rows = []
    n_cycles_matched = 0
    n_variants = 0
    variant_counts_by_soc = {s: 0 for s in SOC_START_POINTS}

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

        n_cycles_matched += 1
        voltage = seg["voltage_volt"].values
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

    status = (f"{cell_id} ({chemistry}): BOL={bol_capacity * 1000:.3f} mAh, "
              f"{len(segments)} discharge cycles scanned, {n_cycles_matched} cycles in "
              f"[{SOH_MIN}, {SOH_MAX}] & C-rate [{C_RATE_MIN}, {C_RATE_MAX}], "
              f"{n_variants} SOC variants kept")
    if not rows:
        return None, status
    return pd.concat(rows, ignore_index=True), status


def main():
    cell_dirs = sorted(d for d in glob.glob(os.path.join(EMPA_DIR, "empa__ccid*")) if os.path.isdir(d))
    print(f"Found {len(cell_dirs)} cell directories in '{EMPA_DIR}'")
    print(f"Target: SOH in [{SOH_MIN}, {SOH_MAX}], C-rate [{C_RATE_MIN}, {C_RATE_MAX}]\n")

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
        print("\nNo matching cycles found anywhere -- widen SOH_MIN/SOH_MAX or the C-rate window.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_RAW_CSV), exist_ok=True)
    combined.to_csv(OUT_RAW_CSV, index=False)

    n_variations = combined["Variation_ID"].nunique()
    print(f"\n{'=' * 70}")
    print(f"-> {OUT_RAW_CSV}")
    print(f"   {n_variations} real discharge cycles kept "
          f"({n_lfp_cells} LFP cells, {n_nmc_cells} NMC cells contributing), "
          f"{len(combined)} rows")
    print(combined.groupby("Chemistry")["Variation_ID"].nunique())


if __name__ == "__main__":
    main()