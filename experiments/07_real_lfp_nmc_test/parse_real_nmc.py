"""
Reshapes a handful of real NMC cycling files from
DownloadedData/Dataset_2_NCM_battery/ (Zhu et al., Nature Communications
2022) into the (Time [s], Voltage [V], Capacity [A.h], Chemistry,
Variation_ID) shape experiments/03_continuous_discharge_soc_sweep/
feature_engineering_continuous.py expects.

Uses one cell per test temperature (25C/35C/45C) to give some real
diversity while keeping scope comparable to the single real LFP cell.
Each file's 'Q discharge/mA.h' column was confirmed to reset to 0 at the
start of every cycle and stay flat outside the real discharge segment, so
-- like the LFP parser -- no explicit phase segmentation is needed;
feature_engineering_continuous.py's own dQ > 1e-5 filter selects the
discharging portion automatically.
"""
import os

import pandas as pd

NMC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "DownloadedData", "Dataset_2_NCM_battery")
NMC_DIR = os.path.normpath(NMC_DIR)

NMC_FILES = ["CY25-05_1-#1.csv", "CY35-05_1-#1.csv", "CY45-05_1-#1.csv"]


def parse_nmc_files(nmc_dir=NMC_DIR, filenames=NMC_FILES):
    print(f"Reading real NMC cell files from '{nmc_dir}'...")
    all_rows = []
    for filename in filenames:
        path = os.path.join(nmc_dir, filename)
        cell_label = filename.replace(".csv", "")
        df = pd.read_csv(path, usecols=["time/s", "Ecell/V", "Q discharge/mA.h", "cycle number"])

        for cycle_number, cycle_df in df.groupby("cycle number"):
            out = pd.DataFrame({
                "Time [s]": cycle_df["time/s"],
                "Voltage [V]": cycle_df["Ecell/V"],
                "Capacity [A.h]": cycle_df["Q discharge/mA.h"] / 1000.0,
                "Chemistry": "NMC",
                "Variation_ID": f"{cell_label}_cycle_{int(cycle_number)}",
                # See parse_real_lfp.py: constant placeholders to satisfy
                # feature_engineering_continuous.py's groupby schema;
                # Variation_ID is already globally unique.
                "Target_Capacity_Ah": 0.0,
                "Size_Multiplier": 0.0,
                "SOH": 0.0,
                "Initial_SOC": 0.0,
            })
            all_rows.append(out)
        print(f"  -> {filename}: {df['cycle number'].nunique()} cycles, {len(df)} rows")

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"  -> total: {len(filenames)} cells, {combined['Variation_ID'].nunique()} cycles, "
          f"{len(combined)} rows, voltage range [{combined['Voltage [V]'].min():.3f}, "
          f"{combined['Voltage [V]'].max():.3f}] V")
    return combined


if __name__ == "__main__":
    df = parse_nmc_files()
    print(df.head())
    print(df.tail())
