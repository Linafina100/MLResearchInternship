"""
Parses the EMPA RO-Crate cycling-aging dataset (data/Dataset-rocrate/,
199 coin cells: 167 NMC / 32 LFP, Battery2030+/PREMISE project) into the
(Time [s], Voltage [V], Capacity [A.h], Chemistry, Variation_ID) shape
feature_engineering.py expects -- selecting only the cycles that are
comparable to this project's established synthetic sim-to-real test
point (SOH=0.8, C-rate 0.1-0.2C; see experiments/21_lfp_diffusivity_tuning/).

Each cell's *.bdf.parquet has one row per ~10s sample across up to ~1400
charge/discharge cycles at fixed ambient_temperature_celsius=25C (every
cell checked was constant 25C -- there is no real temperature sweep in
this dataset, unlike the synthetic side's 15-35C randomization; 25C
falls inside that range so this is not a domain mismatch, just a
narrower real slice than the synthetic training distribution).

Per cycle: current_ampere > 0 is charge (voltage rises to ~4.2V/3.6V),
current_ampere < 0 is discharge (voltage falls to a low cutoff) --
confirmed by inspection. Only the discharge segment of each cycle is
kept. Capacity has no direct column here (unlike exp07's real LFP/NMC
sources) so it's derived by cumulative trapezoidal integration of
|current| over time within the discharge segment.

SOH and C-rate aren't given either -- both are derived empirically per
cell: BOL (beginning-of-life) capacity is the max discharge capacity
over that cell's first 10 cycles (formation cycles included), SOH(n) =
capacity(n)/BOL, C-rate(n) = mean|current(n)| / BOL. Cycles are kept
only if SOH falls in SOH_TOLERANCE around SOH_TARGET and C-rate falls
in [C_RATE_MIN, C_RATE_MAX].
"""
import glob
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ROCRATE_DIR = os.path.join(PROJECT_DIR, "data", "Dataset-rocrate")

SOH_TARGET = float(os.environ.get("SOH_TARGET", 0.8))
SOH_TOLERANCE = float(os.environ.get("SOH_TOLERANCE", 0.02))
# NOTE: unlike the synthetic side, this real dataset does NOT offer a
# 0.1-0.2C option for every chemistry -- diagnosed empirically (see
# RESULTS.md): all 32 real LFP cells were cycled at a fixed ~1.0-1.07C,
# while the 167 real NMC cells span a wide range (~0.07C-1.3C, a genuine
# minority near 0.1-0.2C). A strict [0.1, 0.2] C-rate filter would
# silently drop LFP from the real test set entirely. Default here is
# "no real C-rate filtering" (use every SOH-matched cycle regardless of
# rate) so both chemistries are represented; run again with
# C_RATE_MIN=0.1 C_RATE_MAX=0.2 for the strict (NMC-only) subset.
C_RATE_MIN = float(os.environ.get("C_RATE_MIN", 0.0))
C_RATE_MAX = float(os.environ.get("C_RATE_MAX", 999.0))
CURRENT_EPS_A = 5e-7  # below this, treat as rest/noise, not charge or discharge
MIN_SEGMENT_POINTS = 15
BOL_WINDOW_CYCLES = 10  # first N cycles used to establish BOL capacity
MAX_SEGMENT_DURATION_HOURS = 24.0  # rejects the occasional multi-week pre-test
                                    # storage hold masquerading as "cycle 0"
                                    # (near-constant tiny current at a flat,
                                    # already-low voltage for 1000+ hours --
                                    # confirmed by inspection, not a real
                                    # discharge)
MIN_VOLTAGE_SPAN_V = 0.3  # a real discharge sweeps a meaningful voltage
                           # range; a flat trace at fixed voltage is a rest
                           # artifact, not a discharge curve
RESCALE_TARGET_AH = 2.0  # These are mAh-scale coin cells (BOL ~1.3-13 mAh)
                          # vs. the synthetic side's Ah-scale cells
                          # (Target_Capacity_Ah in {1.2, 2.0, 3.5}, see
                          # experiments/21.../simulate_batteries_lfp_ocp_tuned.py).
                          # dV/dQ's denominator (Ah) makes its magnitude
                          # scale with absolute cell capacity, not just
                          # chemistry shape -- comparing raw real Ah
                          # against raw synthetic Ah produced a spurious
                          # ~1000x magnitude gap that swamped the real
                          # chemistry signal entirely (verified: model
                          # collapsed to predicting NMC for every real
                          # sample). Each real cell's capacity trace is
                          # rescaled by capacity_fraction * RESCALE_TARGET_AH
                          # (mapping its own [0, BOL] onto [0, 2.0] Ah) so
                          # dV/dQ is computed on the same absolute capacity
                          # footing the synthetic model was trained on --
                          # equivalent to comparing at matched SOC, not
                          # matched raw charge.
N_RESAMPLE_POINTS = 80  # feature_engineering.py's create_features_by_voltage_bins
                         # only counts a dV/dQ transition "valid" if its
                         # step has dQ > 1e-5 Ah -- tuned for the synthetic
                         # side's ~Ah-scale batteries over ~3000 t_interp
                         # points. These real coin cells hold ~1-13 mAh
                         # total, natively sampled every 10s (~300-1000
                         # points/segment): raw per-step dQ is ~1e-6 Ah,
                         # an order of magnitude under the threshold, so
                         # nearly every real battery was silently dropped
                         # until this resample was added. Interpolating
                         # onto a coarse, evenly-spaced-in-capacity grid
                         # keeps per-step dQ safely above 1e-5 Ah even for
                         # the smallest (~1.3 mAh) real cells.

ACTIVE_MATERIAL_TO_CHEMISTRY = {
    "LithiumIronPhosphateOxide": "LFP",
    "LithiumNickelCobaltManganeseOxide": "NMC",
}

RUN_LABEL = os.environ.get("RUN_LABEL", "22_real_empa_soh_0.8")
OUT_RAW_CSV = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "raw", "real_empa_raw.csv")


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
    """Yields (cycle_number, segment_df) for each cycle's discharge phase
    (current_ampere < -CURRENT_EPS_A), sorted by time within the cycle."""
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
    return t_hours.values * 3600.0, cap  # (time_seconds_from_segment_start, capacity_ah)


def resample_on_capacity(t_s, voltage, cap_ah, n_points=N_RESAMPLE_POINTS):
    """Interpolates (Time, Voltage) onto n_points evenly spaced along the
    monotonic Capacity axis -- see N_RESAMPLE_POINTS for why."""
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

    # BOL capacity from the first BOL_WINDOW_CYCLES cycles.
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
    n_matched = 0
    for cycle_number, seg in segments:
        t_s, cap_ah = segment_capacity_ah(seg)
        capacity = cap_ah[-1]
        if capacity <= 0:
            continue
        soh = capacity / bol_capacity
        mean_current_a = seg["current_ampere"].abs().mean()
        c_rate = mean_current_a / bol_capacity

        if abs(soh - SOH_TARGET) > SOH_TOLERANCE:
            continue
        if not (C_RATE_MIN <= c_rate <= C_RATE_MAX):
            continue

        n_matched += 1
        cap_ah_rescaled = cap_ah / bol_capacity * RESCALE_TARGET_AH
        t_rs, v_rs, cap_rs = resample_on_capacity(t_s, seg["voltage_volt"].values, cap_ah_rescaled)
        rows.append(pd.DataFrame({
            "Time [s]": t_rs,
            "Voltage [V]": v_rs,
            "Capacity [A.h]": cap_rs,
            "Chemistry": chemistry,
            "Variation_ID": f"{cell_id}_cycle_{cycle_number}",
            "Target_Capacity_Ah": RESCALE_TARGET_AH,
            "Real_BOL_Capacity_mAh": bol_capacity * 1000,
            "Size_Multiplier": 0.0,
            "SOH": round(float(soh), 4),
            "Initial_SOC": 1.0,
            "Ambient_Temperature_C": float(seg["ambient_temperature_celsius"].mean()),
        }))

    status = (f"{cell_id} ({chemistry}): BOL={bol_capacity * 1000:.3f} mAh, "
              f"{len(segments)} discharge cycles scanned, {n_matched} matched "
              f"SOH={SOH_TARGET}+/-{SOH_TOLERANCE} & C-rate [{C_RATE_MIN},{C_RATE_MAX}]")
    if not rows:
        return None, status
    return pd.concat(rows, ignore_index=True), status


def main():
    cell_dirs = sorted(d for d in glob.glob(os.path.join(ROCRATE_DIR, "empa__ccid*")) if os.path.isdir(d))
    print(f"Found {len(cell_dirs)} cell directories in '{ROCRATE_DIR}'")
    print(f"Target: SOH={SOH_TARGET}+/-{SOH_TOLERANCE}, C-rate [{C_RATE_MIN}, {C_RATE_MAX}]\n")

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
        print("\nNo matching cycles found anywhere -- widen SOH_TOLERANCE or the C-rate window.")
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
