import os
import glob
import pandas as pd
import numpy as np

# Direct import of the shared truncation function
from soc_truncation import truncate_at_soc

# --- PATHS & CONFIGURATION ---
PROJECT_DIR = "/Users/ellalindgren/Documents/MLResearchInternship"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "UU_data")
NMC_DIR = os.path.join(DATA_DIR, "NMC")
LFP_DIR = os.path.join(DATA_DIR, "LFP")
OUT_RAW_CSV = os.path.join(DATA_DIR, "combined_uu_experimental_data.csv")

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
MIN_RAW_POINTS_AFTER_TRUNCATION = 5
RESCALE_TARGET_AH = 2.0
N_RESAMPLE_POINTS = 80
CURRENT_EPS_A = 5e-7  # in Amps (or converted appropriately)


def resample_on_capacity(t_s, voltage, cap_ah, n_points=N_RESAMPLE_POINTS):
    cap_grid = np.linspace(cap_ah[0], cap_ah[-1], n_points)
    t_resampled = np.interp(cap_grid, cap_ah, t_s)
    v_resampled = np.interp(cap_grid, cap_ah, voltage)
    return t_resampled, v_resampled, cap_grid


def segment_capacity_ah(t_sec, i_amp):
    t_hours = t_sec / 3600.0
    i_abs = np.abs(i_amp)
    cap = np.concatenate([[0.0], np.cumsum(
        np.diff(t_hours) * (i_abs[:-1] + i_abs[1:]) / 2.0
    )])
    return t_sec, cap

def process_nmc_file(fpath):
    cell_id = os.path.splitext(os.path.basename(fpath))[0]
    try:
        # Read actual experimental measurements from the record sheet
        df = pd.read_excel(fpath, sheet_name="record", header=0)

        col_map = {}

        for c in df.columns:
            c_str = str(c).strip().lower()

            # Match Time exactly, not Total Time
            if c_str == "time":
                col_map[c] = "time"
            elif "current" in c_str:
                col_map[c] = "current_ma"
            elif "voltage" in c_str:
                col_map[c] = "voltage_v"
            elif "temperature" in c_str:
                col_map[c] = "temperature_c"

        df = df.rename(columns=col_map)

        if not {"time", "current_ma", "voltage_v"}.issubset(df.columns):
            print(f"NMC {cell_id} missing columns: {list(df.columns)}")
            return None

        df["current_ampere"] = pd.to_numeric(df["current_ma"], errors="coerce") / 1000.0
        df["voltage_volt"] = pd.to_numeric(df["voltage_v"], errors="coerce")
        if "temperature_c" in df.columns:
            df["temperature_celsius"] = pd.to_numeric(df["temperature_c"], errors="coerce")

        def parse_time_to_seconds(t_val):
            if pd.isna(t_val):
                return np.nan
            if isinstance(t_val, (int, float)):
                return float(t_val)
            t_str = str(t_val).strip()
            parts = t_str.split(":")
            try:
                if len(parts) == 3:
                    return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
                elif len(parts) == 2:
                    return float(parts[0]) * 60.0 + float(parts[1])
                else:
                    return float(t_str)
            except:
                return np.nan

        df["time_sec"] = df["time"].apply(parse_time_to_seconds)
        df = df.dropna(subset=["time_sec", "current_ampere", "voltage_volt"])
        if df.empty:
            return None

        all_rows = []

        for (cycle_num, step_num), cycle_df in df.groupby(
            ["Cycle Index", "Step Index"], sort=False
        ):
            # Only process discharge steps
            if not cycle_df["Step Type"].astype(str).str.contains(
                "DChg", case=False
            ).any():
                continue

            cycle_df = cycle_df.sort_values("time_sec", kind="stable")

            seg = cycle_df[
                cycle_df["current_ampere"] < -0.0005
            ]

            if len(seg) < MIN_RAW_POINTS_AFTER_TRUNCATION:
                continue

            t_s = seg["time_sec"].values - seg["time_sec"].values[0]
            voltage = seg["voltage_volt"].values
            current = seg["current_ampere"].values

            t_s, cap_ah = segment_capacity_ah(t_s, current)
            if len(cap_ah) == 0 or cap_ah[-1] <= 0:
                continue

            ambient_t = float(seg["temperature_celsius"].mean()) if "temperature_celsius" in seg and not seg["temperature_celsius"].isna().all() else 25.0

            for s in SOC_START_POINTS:
                truncated = truncate_at_soc(t_s, voltage, cap_ah, s)
                if truncated is None:
                    continue
                t_trunc, v_trunc, cap_trunc = truncated
                t_rs, v_rs, cap_rs = resample_on_capacity(t_trunc, v_trunc, cap_trunc)

                var_id = f"{cell_id}_cycle_{cycle_num}_soc_{s}"
                all_rows.append(pd.DataFrame({
                    "Time [s]": t_rs,
                    "Voltage [V]": v_rs,
                    "Capacity [A.h]": cap_rs,
                    "Chemistry": "NMC",
                    "DataKind": "real",
                    "Battery_ID": var_id,
                    "Variation_ID": var_id,
                    "Target_Capacity_Ah": RESCALE_TARGET_AH,
                    "Size_Multiplier": 1.0,
                    "SOH": 1.0,
                    "Initial_SOC": s,
                    "Ambient_Temperature_C": ambient_t,
                }))

        return pd.concat(all_rows, ignore_index=True) if all_rows else None
    except Exception as e:
        print(f"Error processing NMC {cell_id}: {e}")
        return None

def process_lfp_file(fpath):
    cell_id = os.path.basename(fpath)
    try:
        df = pd.read_parquet(fpath)
        # LFP columns: ['Test Time / s', 'Voltage / V', 'Current / A', 'Cycle Count / 1', ...]
        required_cols = ['Test Time / s', 'Voltage / V', 'Current / A', 'Cycle Count / 1']
        if not all(col in df.columns for col in required_cols):
            return None

        all_rows = []
        for cycle_num, cycle_df in df.groupby('Cycle Count / 1'):
            cycle_df = cycle_df.sort_values('Test Time / s', kind="stable")
            seg = cycle_df[cycle_df['Current / A'] < -CURRENT_EPS_A]
            if len(seg) < MIN_RAW_POINTS_AFTER_TRUNCATION:
                continue

            t_s = seg['Test Time / s'].values - seg['Test Time / s'].values[0]
            voltage = seg['Voltage / V'].values
            current = seg['Current / A'].values

            t_s, cap_ah = segment_capacity_ah(t_s, current)
            if len(cap_ah) == 0 or cap_ah[-1] <= 0:
                continue

            ambient_t = 25.0  # Default if not present in parquet

            for s in SOC_START_POINTS:
                truncated = truncate_at_soc(t_s, voltage, cap_ah, s)
                if truncated is None:
                    continue
                t_trunc, v_trunc, cap_trunc = truncated
                t_rs, v_rs, cap_rs = resample_on_capacity(t_trunc, v_trunc, cap_trunc)

                var_id = f"{cell_id}_cycle_{cycle_num}_soc_{s}"
                all_rows.append(pd.DataFrame({
                    "Time [s]": t_rs,
                    "Voltage [V]": v_rs,
                    "Capacity [A.h]": cap_rs,
                    "Chemistry": "LFP",
                    "DataKind": "real",
                    "Battery_ID": var_id,
                    "Variation_ID": var_id,
                    "Target_Capacity_Ah": RESCALE_TARGET_AH,
                    "Size_Multiplier": 1.0,
                    "SOH": 1.0,
                    "Initial_SOC": s,
                    "Ambient_Temperature_C": ambient_t,
                }))

        return pd.concat(all_rows, ignore_index=True) if all_rows else None
    except Exception as e:
        print(f"Error processing LFP {cell_id}: {e}")
        return None
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    nmc_files = sorted(glob.glob(os.path.join(NMC_DIR, "*.xlsx")))
    lfp_files = sorted(glob.glob(os.path.join(LFP_DIR, "*.parquet")))
    
    print(f"Found {len(nmc_files)} NMC files and {len(lfp_files)} LFP files.")
    
    frames = []
    print("\nProcessing NMC files...")
    for idx, fpath in enumerate(nmc_files, 1):
        res = process_nmc_file(fpath)
        if res is not None:
            frames.append(res)
            
    print("\nProcessing LFP files...")
    for idx, fpath in enumerate(lfp_files, 1):
        res = process_lfp_file(fpath)
        if res is not None:
            frames.append(res)

    if not frames:
        print("\nError: No valid data processed. Check file paths and columns.")
        return

    # 1. Save your combined UU experimental data
    combined_df = pd.concat(frames, ignore_index=True)
    combined_df.to_csv(OUT_RAW_CSV, index=False)
    print(f"\nSuccessfully created combined multi-SOC experimental dataset at: {OUT_RAW_CSV}")
    print(f"Total rows: {len(combined_df):,}")
    print(combined_df.groupby("Chemistry")["Variation_ID"].nunique())
    
    # 2. MERGE with synthetic data and write to outputs/sim_and_real_raw.csv (just like EMPA!)
    synthetic_path = os.path.join(DATA_DIR, "advanced_synthetic_battery_data.csv")
    if os.path.exists(synthetic_path):
        print("Merging real UU data with synthetic data for evaluation cache...")
        df_synth = pd.read_csv(synthetic_path)
        # Ensure synthetic data has DataKind set correctly
        if "DataKind" not in df_synth.columns:
            df_synth["DataKind"] = "synthetic"
            
        sim_and_real = pd.concat([combined_df, df_synth], ignore_index=True)
    else:
        sim_and_real = combined_df

    outputs_sim_real_path = os.path.join(PROJECT_DIR, "UU_pipeline", "outputs", "sim_and_real_raw.csv")
    sim_and_real.to_csv(outputs_sim_real_path, index=False)
    
    print(f"Successfully cached combined sim+real dataset at: {outputs_sim_real_path}")


if __name__ == "__main__":
    main()