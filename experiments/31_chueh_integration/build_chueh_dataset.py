"""
Parses the Severson et al. (2019) / Chueh lab LFP batch MATLAB struct files
(data/*_batchdata_updated_struct_errorcorrect.mat, 124 LFP cells) into the
(Time [s], Voltage [V], Capacity [A.h], Chemistry, Variation_ID) shape
feature_engineering.py expects -- selecting only cycles matching the target
SOH (default 0.8) and C-rate window.

All cells in this dataset are LFP (A123 APR18650M1A, 1.1 Ah nominal capacity)
cycled at a fixed 4C discharge rate down to 2.0V under various fast-charging
policies.

Mirrors the architecture and scaling strategy of build_empa_dataset.py:
- Extracts discharge segments (current < -CURRENT_EPS_A).
- Computes empirical BOL capacity from the first BOL_WINDOW_CYCLES.
- Rescales capacity to RESCALE_TARGET_AH = 2.0 to prevent the dV/dQ
  magnitude mismatch against the synthetic training data.
- Resamples onto an evenly spaced capacity grid (N_RESAMPLE_POINTS = 80).
"""
import glob
import os
import h5py
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "31_chueh_data")

SOH_TARGET = float(os.environ.get("SOH_TARGET", 0.8))
SOH_TOLERANCE = float(os.environ.get("SOH_TOLERANCE", 0.02))
C_RATE_MIN = float(os.environ.get("C_RATE_MIN", 3.5))  # Severson LFP discharges at 4C (~3.6C-4.4C effective)
C_RATE_MAX = float(os.environ.get("C_RATE_MAX", 4.5))
CURRENT_EPS_A = 0.1  # 4C discharge is ~4.4A; anything below 0.1A is rest/charge
MIN_SEGMENT_POINTS = 15
BOL_WINDOW_CYCLES = 5  # first N cycles used to establish BOL capacity
MIN_VOLTAGE_SPAN_V = 1.0  # LFP full discharge spans ~3.3V down to 2.0V (~1.3V span)
RESCALE_TARGET_AH = 2.0   | Matches the synthetic training scale to align dV/dQ denominators
N_RESAMPLE_POINTS = 80

RUN_LABEL = os.environ.get("RUN_LABEL", "31_chueh_integration")
OUT_RAW_CSV = os.path.join(PROJECT_DIR, "data", RUN_LABEL, "raw", "chueh_raw.csv")


def dereference_h5_ref(f, ref):
    """Helper to dereference h5py object reference pointers in MATLAB v7.3 files."""
    if hasattr(ref, 'shape') and ref.shape == ():
        return f[ref]
    return ref


def extract_cycle_data(f, cycle_group):
    """Extracts time, voltage, and current arrays from a MATLAB cycle struct group."""
    try:
        t_ref = cycle_group['t']
        v_ref = cycle_group['V']
        i_ref = cycle_group['I']
        
        t = np.array(dereference_h5_ref(f, t_ref)).flatten()
        v = np.array(dereference_h5_ref(f, v_ref)).flatten()
        i = np.array(dereference_h5_ref(f, i_ref)).flatten()
        return t, v, i
    except Exception:
        return None, None, None


def discharge_segments_from_cycle(t, v, i):
    """Filters cycle arrays for the discharge phase (current < -CURRENT_EPS_A)."""
    mask = i < -CURRENT_EPS_A
    if mask.sum() < MIN_SEGMENT_POINTS:
        return None
    
    seg_t = t[mask]
    seg_v = v[mask]
    seg_i = i[mask]
    
    # Sort chronologically just in case
    sort_idx = np.argsort(seg_t)
    seg_t = seg_t[sort_idx] - seg_t[sort_idx][0]  # start time at 0
    seg_v = seg_v[sort_idx]
    seg_i = seg_i[sort_idx]
    
    voltage_span = seg_v.max() - seg_v.min()
    if voltage_span < MIN_VOLTAGE_SPAN_V:
        return None
        
    seg_df = pd.DataFrame({
        "test_time_second": seg_t,
        "voltage_volt": seg_v,
        "current_ampere": seg_i
    })
    return seg_df


def segment_capacity_ah(seg):
    t_hours = seg["test_time_second"].values / 3600.0
    i_abs = seg["current_ampere"].abs().values
    if len(t_hours) < 2:
        return t_hours * 3600.0, np.zeros_like(t_hours)
    cap = np.concatenate([[0.0], np.cumsum(
        np.diff(t_hours) * (i_abs[:-1] + i_abs[1:]) / 2.0
    )])
    return seg["test_time_second"].values, cap


def resample_on_capacity(t_s, voltage, cap_ah, n_points=N_RESAMPLE_POINTS):
    if cap_ah[-1] <= cap_ah[0]:
        cap_ah = cap_ah + np.linspace(0, 1e-6, len(cap_ah))
    cap_grid = np.linspace(cap_ah[0], cap_ah[-1], n_points)
    t_resampled = np.interp(cap_grid, cap_ah, t_s)
    v_resampled = np.interp(cap_grid, cap_ah, voltage)
    return t_resampled, v_resampled, cap_grid


def process_batch_file(mat_path):
    print(f"\nProcessing batch file: {os.path.basename(mat_path)}...")
    f = h5py.File(mat_path, 'r')
    
    if 'batch' not in f:
        print(f"Error: 'batch' group not found in {mat_path}")
        return []

    batch_group = f['batch']
    cell_names = list(batch_group.keys())
    print(f"  Found {len(cell_names)} cells in batch struct.")

    batch_rows = []
    n_matched_total = 0

    for cell_idx, cell_key in enumerate(cell_names):
        try:
            cell_group = batch_group[cell_key]
            if 'cycles' not in cell_group:
                continue
            
            cycles_group = cell_group['cycles']
            cycle_keys = list(cycles_group.keys())
            
            # First pass: collect valid discharge capacities to establish BOL capacity
            cycle_data_cache = []
            for cyc_idx, cyc_key in enumerate(cycle_keys):
                cyc_group = cycles_group[cyc_key]
                t, v, i = extract_cycle_data(f, cyc_group)
                if t is None:
                    continue
                seg_df = discharge_segments_from_cycle(t, v, i)
                if seg_df is None:
                    continue
                t_s, cap_ah = segment_capacity_ah(seg_df)
                capacity = cap_ah[-1]
                if capacity > 0:
                    cycle_data_cache.append((cyc_idx, seg_df, t_s, cap_ah, capacity))

            if not cycle_data_cache:
                continue

            # Establish BOL capacity from early cycles
            early_caps = [cap for idx, _, _, _, cap in cycle_data_cache if idx < BOL_WINDOW_CYCLES]
            if not early_caps:
                early_caps = [cap for _, _, _, _, cap in cycle_data_cache[:BOL_WINDOW_CYCLES]]
            bol_capacity = max(early_caps) if early_caps else 1.1

            n_matched_cell = 0
            for cyc_idx, seg_df, t_s, cap_ah, capacity in cycle_data_cache:
                soh = capacity / bol_capacity
                mean_current = seg_df["current_ampere"].abs().mean()
                c_rate = mean_current / bol_capacity

                if abs(soh - SOH_TARGET) > SOH_TOLERANCE:
                    continue
                if not (C_RATE_MIN <= c_rate <= C_RATE_MAX):
                    continue

                n_matched_cell += 1
                cap_ah_rescaled = cap_ah / bol_capacity * RESCALE_TARGET_AH
                t_rs, v_rs, cap_rs = resample_on_capacity(t_s, seg_df["voltage_volt"].values, cap_ah_rescaled)

                batch_rows.append(pd.DataFrame({
                    "Time [s]": t_rs,
                    "Voltage [V]": v_rs,
                    "Capacity [A.h]": cap_rs,
                    "Chemistry": "LFP",
                    "Variation_ID": f"{os.path.basename(mat_path)}_{cell_key}_cycle_{cyc_idx}",
                    "Target_Capacity_Ah": RESCALE_TARGET_AH,
                    "Real_BOL_Capacity_Ah": bol_capacity,
                    "Size_Multiplier": 0.0,
                    "SOH": round(float(soh), 4),
                    "Initial_SOC": 1.0,
                    "Ambient_Temperature_C": 30.0,  # Severson chamber set point
                }))

            n_matched_total += n_matched_cell
        except Exception as e:
            print(f"  Warning: Skipping cell {cell_key} due to error: {e}")
            continue

    f.close()
    print(f"  -> Extracted {n_matched_total} matching cycles from this batch.")
    return batch_rows


def main():
    # Find all batch mat files in the data directory
    mat_pattern = os.path.join(DATA_DIR, "*_batchdata_updated_struct_errorcorrect.mat")
    mat_files = sorted(glob.glob(mat_pattern))

    if not mat_files:
        print(f"Error: No MATLAB batch files found matching pattern: {mat_pattern}")
        print("Please place your downloaded Severson batch .mat files into the data/ directory.")
        return

    print(f"Found {len(mat_files)} batch MATLAB files.")
    print(f"Target: SOH={SOH_TARGET}+/-{SOH_TOLERANCE}, C-rate [{C_RATE_MIN}, {C_RATE_MAX}]")

    all_rows = []
    for mat_path in mat_files:
        batch_rows = process_batch_file(mat_path)
        all_rows.extend(batch_rows)

    if not all_rows:
        print("\nNo matching cycles found across any batch files. Check SOH/C-rate thresholds.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_RAW_CSV), exist_ok=True)
    combined.to_csv(OUT_RAW_CSV, index=False)

    n_variations = combined["Variation_ID"].nunique()
    print(f"\n{'=' * 70}")
    print(f"-> Saved combined Chueh raw data to: {OUT_RAW_CV if 'OUT_RAW_CV' in locals() else OUT_RAW_CSV}")
    print(f"   {n_variations} LFP discharge cycles kept across all batches, {len(combined)} total rows.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()