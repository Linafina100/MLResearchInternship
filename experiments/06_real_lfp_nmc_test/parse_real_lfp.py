"""
Reshapes the real LFP discharge data in DownloadedData/ENSAYOSBATERIA/
(a single LiFePO4 32700, 6000mAh cell, 626 charge/discharge cycles) into
the (Time [s], Voltage [V], Capacity [A.h], Chemistry, Variation_ID)
shape experiments/03_continuous_discharge_soc_sweep/feature_engineering_continuous.py
expects.

Each DESCARGA<n>.txt file (per readme.txt) has 5 undocumented-header,
tab-separated columns -- Voltage [V], charge current [A], discharge
current [A], Temperature [C], timestamp [s] -- and already contains a
full rest -> 6A discharge (2.5V cutoff) -> rest cycle in one file. No
phase segmentation is needed: Capacity [A.h] is derived by cumulatively
integrating the discharge current over time, which is flat (zero slope)
during both rest segments and only rises during the real discharge --
feature_engineering_continuous.py's own dQ > 1e-5 validity filter then
naturally selects just the discharging portion.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

DESCARGA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "DownloadedData", "ENSAYOSBATERIA", "descargasVF")
DESCARGA_DIR = os.path.normpath(DESCARGA_DIR)

COLUMNS = ["Voltage [V]", "I_charge_A", "I_discharge_A", "Temperature_C", "Timestamp_s"]


def parse_lfp_discharge_files(descarga_dir=DESCARGA_DIR):
    print(f"Reading real LFP discharge files from '{descarga_dir}'...")
    files = sorted(glob.glob(os.path.join(descarga_dir, "DESCARGA*.txt")),
                    key=lambda p: int(re.search(r"DESCARGA(\d+)\.txt", p).group(1)))

    all_rows = []
    for path in files:
        cycle_number = int(re.search(r"DESCARGA(\d+)\.txt", path).group(1))
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)

        # Capacity via cumulative trapezoidal integration of discharge
        # current over time (flat during both rest segments, rising only
        # during the real 6A discharge).
        t_hours = (df["Timestamp_s"] - df["Timestamp_s"].iloc[0]) / 3600.0
        capacity_ah = np.concatenate([[0.0], np.cumsum(
            np.diff(t_hours) * (df["I_discharge_A"].values[:-1] + df["I_discharge_A"].values[1:]) / 2.0
        )])

        out = pd.DataFrame({
            "Time [s]": df["Timestamp_s"] - df["Timestamp_s"].iloc[0],
            "Voltage [V]": df["Voltage [V]"],
            "Capacity [A.h]": capacity_ah,
            "Chemistry": "LFP",
            "Variation_ID": f"lfp_cycle_{cycle_number}",
            # feature_engineering_continuous.py's groupby key includes these
            # synthetic-only metadata columns; Variation_ID is already
            # globally unique here, so constant placeholders are enough to
            # satisfy the schema without affecting grouping. ml_pipeline.py
            # excludes all of these from the actual feature matrix anyway.
            "Target_Capacity_Ah": 0.0,
            "Size_Multiplier": 0.0,
            "SOH": 0.0,
            "Initial_SOC": 0.0,
        })
        all_rows.append(out)

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"  -> {len(files)} cycle files, {len(combined)} total rows, "
          f"voltage range [{combined['Voltage [V]'].min():.3f}, {combined['Voltage [V]'].max():.3f}] V")
    return combined


if __name__ == "__main__":
    df = parse_lfp_discharge_files()
    print(df.head())
    print(df.tail())
