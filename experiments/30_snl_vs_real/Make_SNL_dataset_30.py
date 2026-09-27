import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Define paths to both folders
DATA_DIR_1 = Path(
    r"C:\Users\hilda\OneDrive\Skrivbord\forskningapraktik\MLResearchInternship\data\SNL LFP"
)
# Update this path with your second folder's location:
DATA_DIR_2 = Path(
    r"C:\Users\hilda\OneDrive\Skrivbord\forskningapraktik\MLResearchInternship\data\SNL NMC"
)

# Destination output file
OUTPUT_FILE = Path(
    r"C:\Users\hilda\OneDrive\Skrivbord\forskningapraktik\MLResearchInternship\data\combined_target_data.csv"
)

TARGET_COLUMNS = [
    "Time [s]",
    "Voltage [V]",
    "Capacity [A.h]",
    "Chemistry",
    "Target_Capacity_Ah",
    "C_Rate",
    "Size_Multiplier",
    "SOH",
    "Initial_SOC",
    "Ambient_Temperature_C",
    "Resistance_Factor",
    "Base_Parameter_Set",
    "V_min [V]",
    "Variation_ID",
]


def parse_metadata_from_filename(filename: str, fallback_chem: str = "LFP") -> dict:
    meta = {
        "Chemistry": fallback_chem,
        "Ambient_Temperature_C": 25.0,
        "C_Rate": 0.5,
        "Initial_SOC": 1.0,
    }

    temp_match = re.search(r"_(\d+)C_", filename)
    if temp_match:
        meta["Ambient_Temperature_C"] = float(temp_match.group(1))

    chem_match = re.search(r"_(LFP|NMC|NCA)_", filename, re.IGNORECASE)
    if chem_match:
        meta["Chemistry"] = chem_match.group(1).upper()

    rate_match = re.search(r"_(\d+(?:\.\d+)?)(?:-\d+(?:\.\d+)?)?C_", filename)
    if rate_match:
        meta["C_Rate"] = float(rate_match.group(1))

    soc_match = re.search(r"_(\d+)-(\d+)_", filename)
    if soc_match:
        meta["Initial_SOC"] = float(soc_match.group(2)) / 100.0

    return meta


def process_directory(data_dir: Path, starting_id: int = 0):
    """Processes all timeseries files in a directory, returning a list of dataframes and the next ID."""
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}", file=sys.stderr)
        return [], starting_id

    all_csvs = sorted(data_dir.glob("*.csv"))
    timeseries_files = [
        f
        for f in all_csvs
        if "_cycle_data" not in f.name and f.name != OUTPUT_FILE.name
    ]

    print(f"Found {len(timeseries_files)} timeseries files in {data_dir.name}.")

    dir_data = []
    current_id = starting_id

    for file_path in timeseries_files:
        df = pd.read_csv(file_path)
        meta = parse_metadata_from_filename(file_path.name)

        time_col = next(
            (
                c
                for c in [
                    "Test_Time (s)",
                    "Time [s]",
                    "Time (s)",
                    "Date_Time",
                    "Time",
                ]
                if c in df.columns
            ),
            None,
        )
        voltage_col = next(
            (
                c
                for c in ["Voltage (V)", "Voltage [V]", "Voltage", "Volt"]
                if c in df.columns
            ),
            None,
        )
        capacity_col = next(
            (
                c
                for c in [
                    "Discharge_Capacity (Ah)",
                    "Capacity [A.h]",
                    "Capacity (Ah)",
                    "Capacity",
                ]
                if c in df.columns
            ),
            None,
        )

        out_df = pd.DataFrame(
            {
                "Time [s]": df[time_col] if time_col else 0.0,
                "Voltage [V]": df[voltage_col] if voltage_col else 0.0,
                "Capacity [A.h]": df[capacity_col] if capacity_col else 0.0,
                "Chemistry": meta["Chemistry"],
                "Target_Capacity_Ah": 1.1,
                "C_Rate": meta["C_Rate"],
                "Size_Multiplier": 1.0,
                "SOH": 1.0,
                "Initial_SOC": meta["Initial_SOC"],
                "Ambient_Temperature_C": meta["Ambient_Temperature_C"],
                "Resistance_Factor": 1.0,
                "Base_Parameter_Set": "SNL_18650",
                "V_min [V]": 2.0,
                "Variation_ID": current_id,
            }
        )

        dir_data.append(out_df[TARGET_COLUMNS])
        current_id += 1

    return dir_data, current_id


def main():
    # Process Folder 1
    data_1, next_id = process_directory(DATA_DIR_1, starting_id=0)

    # Process Folder 2 continuing Variation_ID counter
    data_2, total_variations = process_directory(DATA_DIR_2, starting_id=next_id)

    combined_list = data_1 + data_2

    if not combined_list:
        print("No files were successfully processed.", file=sys.stderr)
        return

    combined_df = pd.concat(combined_list, ignore_index=True)

    # Imputation and interpolation
    numeric_cols = ["Time [s]", "Voltage [V]", "Capacity [A.h]"]
    combined_df[numeric_cols] = combined_df.groupby("Variation_ID")[
        numeric_cols
    ].transform(lambda g: g.ffill().bfill())

    imputer = SimpleImputer(strategy="median")
    combined_df[numeric_cols] = imputer.fit_transform(
        combined_df[numeric_cols]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone! Successfully merged {total_variations} variations.")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Exported to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()