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

# Path setups
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)  # For ml_pipeline and feature_engineering

from feature_engineering import create_features_by_voltage_bins
from ml_pipeline import run_ml_pipeline

# --------------------------------------------------------------------------
# Directories and Paths
# --------------------------------------------------------------------------
# Paths to your raw Sandia datasets
SNL_LFP_DIR = Path(
    r"C:\Users\hilda\OneDrive\Skrivbord\forskningapraktik\MLResearchInternship\data\SNL LFP"
)
SNL_NMC_DIR = Path(
    r"C:\Users\hilda\OneDrive\Skrivbord\forskningapraktik\MLResearchInternship\data\SNL NMC"
)

# Output paths for merged data and features
DATA_OUTPUT_DIR = Path(PROJECT_DIR) / "data" / "24_SNL_Real"
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_RAW_CSV = (
    Path(PROJECT_DIR)
    / "data"
    / "24_SNL_1"
    / "raw"
    / "synthetic_battery_data_24.csv"
)
REAL_RAW_CSV = DATA_OUTPUT_DIR / "real_snl_combined_raw.csv"
RAW_CSV = Path(SCRIPT_DIR) / "sim_and_real_raw.csv"
FEATURES_DIR = Path(SCRIPT_DIR) / "features"

GROUPBY_COLS = [
    "Chemistry",
    "Size_Multiplier",
    "SOH",
    "Initial_SOC",
    "Variation_ID",
]
TARGET_ZONE_MIN, TARGET_ZONE_MAX = 2.5, 3.0
BIN_COL_RE = re.compile(r"dV_dQ_V_([\d.]+)_([\d.]+)")

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


# --------------------------------------------------------------------------
# Step 1: Sandia Data Ingestion & Preprocessing (Discharge Filtering)
# --------------------------------------------------------------------------
def parse_metadata_from_filename(
    filename: str, fallback_chem: str = "LFP"
) -> dict:
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


def process_snl_directory(
    data_dir: Path, fallback_chem: str, starting_id: int = 0
):
    """Parses raw Sandia timeseries files, extracting only the valid discharge cycles."""
    if not data_dir.exists():
        print(f"Warning: Directory not found: {data_dir}", file=sys.stderr)
        return [], starting_id

    all_csvs = sorted(data_dir.glob("*.csv"))
    # Exclude cycle summaries
    timeseries_files = [
        f
        for f in all_csvs
        if "_cycle_data" not in f.name and not f.name.startswith("combined_")
    ]

    print(
        f"Processing {len(timeseries_files)} files from {data_dir.name} ({fallback_chem})..."
    )

    processed_dfs = []
    current_id = starting_id

    for file_path in timeseries_files:
        df = pd.read_csv(file_path)
        meta = parse_metadata_from_filename(file_path.name, fallback_chem)

        # Detect columns
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
        current_col = next(
            (
                c
                for c in ["Current (A)", "Current [A]", "Current", "Amps"]
                if c in df.columns
            ),
            None,
        )

        if not time_col or not voltage_col or not capacity_col:
            continue

        # Isolate Discharge Phase (Negative current or positive discharge capacity)
        sub_df = df.copy()
        if current_col:
            # Sandia discharge current is typically negative (< -0.05 A)
            discharge_mask = sub_df[current_col] < -0.05
            if discharge_mask.sum() > 50:
                sub_df = sub_df[discharge_mask]

        # Ensure voltage spans the active battery operating zone
        if sub_df[voltage_col].max() < 2.5:
            continue

        out_df = pd.DataFrame(
            {
                "Time [s]": sub_df[time_col],
                "Voltage [V]": sub_df[voltage_col],
                "Capacity [A.h]": sub_df[capacity_col].abs(),
                "Chemistry": meta["Chemistry"],
                "Target_Capacity_Ah": 1.1 if meta["Chemistry"] == "LFP" else 3.0,
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

        # Forward/Backward fill per cycle
        numeric_cols = ["Time [s]", "Voltage [V]", "Capacity [A.h]"]
        out_df[numeric_cols] = out_df[numeric_cols].ffill().bfill()

        processed_dfs.append(out_df[TARGET_COLUMNS])
        current_id += 1

    return processed_dfs, current_id


def build_real_snl_raw_dataset():
    """Merges LFP and NMC raw datasets and creates the real test CSV."""
    lfp_dfs, next_id = process_snl_directory(
        SNL_LFP_DIR, fallback_chem="LFP", starting_id=0
    )
    nmc_dfs, total_ids = process_snl_directory(
        SNL_NMC_DIR, fallback_chem="NMC", starting_id=next_id
    )

    all_dfs = lfp_dfs + nmc_dfs
    if not all_dfs:
        raise RuntimeError(
            "No Sandia timeseries files could be processed. Check directory paths."
        )

    real_df = pd.concat(all_dfs, ignore_index=True)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    num_cols = ["Time [s]", "Voltage [V]", "Capacity [A.h]"]
    real_df[num_cols] = imputer.fit_transform(real_df[num_cols])

    real_df.to_csv(REAL_RAW_CSV, index=False)
    print(
        f"Saved {len(real_df):,} real timeseries points across {total_ids} variations to '{REAL_RAW_CSV}'"
    )
    return real_df


# --------------------------------------------------------------------------
# Step 2: Feature Engineering & Extraction
# --------------------------------------------------------------------------
def build_combined_raw_csv():
    if not os.path.exists(REAL_RAW_CSV):
        build_real_snl_raw_dataset()

    print(f"Loading synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"

    print(f"Loading real Sandia data from '{REAL_RAW_CSV}'...")
    real_df = pd.read_csv(REAL_RAW_CSV, low_memory=False)
    real_df["DataKind"] = "real"

    combined = pd.concat([sim_df, real_df], ignore_index=True)
    combined.to_csv(RAW_CSV, index=False)
    print(
        f"Combined raw data saved: {(combined['DataKind'] == 'synthetic').sum()} synth, "
        f"{(combined['DataKind'] == 'real').sum()} real rows."
    )
    return combined


def target_zone_bin_cols(all_bin_cols):
    kept = []
    for col in all_bin_cols:
        m = BIN_COL_RE.match(col)
        if m:
            bin_high, bin_low = float(m.group(1)), float(m.group(2))
            if bin_low >= TARGET_ZONE_MIN and bin_high <= TARGET_ZONE_MAX:
                kept.append(col)
    return kept


def extract_features_with_kind():
    if not os.path.exists(RAW_CSV):
        build_combined_raw_csv()
    else:
        print(f"Note: Reusing existing '{RAW_CSV}'.")

    print("\nExtracting features using voltage bins...")
    features_df = create_features_by_voltage_bins(
        str(RAW_CSV), output_dir=str(FEATURES_DIR), exclude_final_transition=True
    )

    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df["Battery_ID"] = raw_df.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = (
        raw_df.drop_duplicates("Battery_ID")
        .set_index("Battery_ID")["DataKind"]
    )
    features_df["DataKind"] = features_df["Battery_ID"].map(battery_to_kind)

    all_bin_cols = [c for c in features_df.columns if c.startswith("dV_dQ_V_")]
    zone_bin_cols = target_zone_bin_cols(all_bin_cols)

    print(f"Kept target-zone ({TARGET_ZONE_MIN}-{TARGET_ZONE_MAX}V) bins: {zone_bin_cols}")
    print(
        features_df.groupby(["DataKind", "Chemistry"])
        .size()
        .rename("n_batteries")
    )
    return features_df, zone_bin_cols


# --------------------------------------------------------------------------
# Step 3: Model Evaluation
# --------------------------------------------------------------------------
def balanced_metrics(synthetic_csv, real_csv, feature_cols):
    synth, real = pd.read_csv(synthetic_csv), pd.read_csv(real_csv)
    X_train = synth[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = real[feature_cols].replace([np.inf, -np.inf], np.nan)

    le = LabelEncoder()
    y_train = le.fit_transform(synth["Chemistry"])
    y_test = le.transform(real["Chemistry"])

    lower, upper = X_train.quantile(0.01), X_train.quantile(0.99)
    X_train_c = X_train.clip(lower=lower, upper=upper, axis=1)
    X_test_c = X_test.clip(lower=lower, upper=upper, axis=1)

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train_c)
    X_test_i = imputer.transform(X_test_c)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)
    X_test_s = scaler.transform(X_test_i)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    out = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        recalls = recall_score(
            y_test, preds, average=None, zero_division=0
        )
        out[name] = {
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "LFP_recall": recalls[0] if len(recalls) > 0 else 0.0,
            "NMC_recall": recalls[1] if len(recalls) > 1 else 0.0,
        }
    return out


def main():
    features_df, zone_bin_cols = extract_features_with_kind()

    if not zone_bin_cols:
        print("Error: No bins survived inside the voltage zone. Exiting.")
        return

    metadata_cols = [
        "Battery_ID",
        "Chemistry",
        "Size_Multiplier",
        "SOH",
        "Initial_SOC",
        "Target_Capacity_Ah",
        "Ambient_Temperature_C",
        "Resistance_Factor",
        "Base_Parameter_Set",
    ]
    keep_cols = [
        c for c in metadata_cols if c in features_df.columns
    ] + zone_bin_cols

    synthetic_csv = FEATURES_DIR / "synthetic_features_target_zone.csv"
    real_csv = FEATURES_DIR / "real_features_target_zone.csv"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    features_df[features_df["DataKind"] == "synthetic"][keep_cols].to_csv(
        synthetic_csv, index=False
    )
    features_df[features_df["DataKind"] == "real"][keep_cols].to_csv(
        real_csv, index=False
    )

    print("\nTraining on synthetic data, evaluating on real Sandia data...")
    results = run_ml_pipeline(
        synthetic_csv=str(synthetic_csv),
        real_csv=str(real_csv),
        save_artifacts=False,
    )
    plt.close("all")

    balanced = balanced_metrics(synthetic_csv, real_csv, zone_bin_cols)

    print("\n=== SUMMARY ===")
    for model_name, raw_acc in results["model_accuracies"].items():
        b = balanced[model_name]
        print(
            f"  {model_name:<15} raw={raw_acc*100:.2f}%  balanced={b['balanced_accuracy']*100:.2f}%  "
            f"LFP_recall={b['LFP_recall']:.3f}  NMC_recall={b['NMC_recall']:.3f}"
        )


if __name__ == "__main__":
    main()
