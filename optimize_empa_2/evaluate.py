"""
Experiment 24: Sim-to-real evaluation för olika Initial_SOC-nivåer (trunkerade kurvor)
med både Random Forest och XGBoost, samt generering av en jämförelsegraf.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Ett steg upp till optimize_empa_2's förälder eller root beroende på struktur

# Peka ut rätt sökväg direkt baserat på var projektet ligger
REAL_RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "raw", "advanced_synthetic_battery_data.csv")

sys.path.insert(0, PROJECT_DIR)

from ml_pipeline import run_ml_pipeline
from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.path.join(PROJECT_DIR, "data", "raw", "advanced_synthetic_battery_data.csv")
REAL_RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']


def extract_features_for_soc(real_raw_path, soc_val):
    """Extraherar eller laddar cachade features för en specifik SOC-nivå. Detta ska snabba på runtimes vid upprepade körningar, som i SOC-sweep."""
    soc_int = int(round(soc_val * 100))
    feature_cache_path = os.path.join(FEATURES_DIR, f"real_features_soc_{soc_int}.csv")
    
    # Om vi redan har beräknat features tidigare, läs in dem direkt på någon sekund!
    if os.path.exists(feature_cache_path):
        print(f"\n--- Laddar cachade features för Initial_SOC = {soc_int}% ---")
        features_df = pd.read_csv(feature_cache_path, low_memory=False)
        all_bin_cols = sorted(
            (c for c in features_df.columns if c.startswith('dV_dQ_V_')),
            key=lambda c: -float(c.split('_')[3]),
        )
        return features_df, all_bin_cols

    print(f"\n--- Extraherar features (första körningen) för Initial_SOC = {soc_int}% ---")
    
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    
    real_df = pd.read_csv(real_raw_path, low_memory=False)
    real_df["DataKind"] = "real"
    
    combined = pd.concat([sim_df, real_df], ignore_index=True)
    temp_raw_csv = os.path.join(SCRIPT_DIR, f"temp_raw_soc_{soc_int}.csv")
    combined.to_csv(temp_raw_csv, index=False)

    features_df = create_features_by_voltage_bins(temp_raw_csv, output_dir=FEATURES_DIR, exclude_final_transition=True)

    raw_df = pd.read_csv(temp_raw_csv, low_memory=False)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    battery_to_kind = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')['DataKind']
    features_df['DataKind'] = features_df['Battery_ID'].map(battery_to_kind)

    all_bin_cols = sorted(
        (c for c in features_df.columns if c.startswith('dV_dQ_V_')),
        key=lambda c: -float(c.split('_')[3]),
    )
    
    # Spara undan i cachen så vi slipper beräkna om nästa gång
    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_df.to_csv(feature_cache_path, index=False)
    
    if os.path.exists(temp_raw_csv):
        os.remove(temp_raw_csv)
        
    return features_df, all_bin_cols


def run_multi_soc_evaluation():
    """Kör träning och utvärdering för varje SOC-nivå separat för RF och XGB."""
    soc_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    
    results = {
        "SOC": [],
        "RF_Balanced_Accuracy": [],
        "XGB_Balanced_Accuracy": []
    }

    print("\n==================================================")
    print(" BÖRJAR MULTI-SOC UTVÄRDERING (RF & XGB)")
    print("==================================================")

    for soc in soc_levels:
        real_csv_path = os.path.join(REAL_RAW_DIR, f"real_empa_raw_soc_{int(soc*100)}.csv")
        if not os.path.exists(real_csv_path):
            print(f"Hoppar över SOC {int(soc*100)}%: Hittade ingen fil på {real_csv_path}")
            continue

        features_df, bin_cols = extract_features_for_soc(real_csv_path, soc)
        if not bin_cols:
            print(f"Inga giltiga bins för SOC {int(soc*100)}%.")
            continue

        metadata_cols = ['Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC']
        keep_cols = [c for c in metadata_cols if c in features_df.columns] + bin_cols

        synth_sub = features_df[features_df['DataKind'] == 'synthetic'][keep_cols].dropna(subset=bin_cols)
        real_sub = features_df[features_df['DataKind'] == 'real'][keep_cols].dropna(subset=bin_cols)

        if len(synth_sub) == 0 or len(real_sub) == 0:
            print(f"För få rader kvar efter rensning för SOC {int(soc*100)}%.")
            continue

        le = LabelEncoder()
        le.fit(pd.concat([synth_sub['Chemistry'], real_sub['Chemistry']]).unique())

        X_train = synth_sub[bin_cols].replace([np.inf, -np.inf], np.nan)
        X_test = real_sub[bin_cols].replace([np.inf, -np.inf], np.nan)

        y_train = le.transform(synth_sub['Chemistry'])
        y_test = le.transform(real_sub['Chemistry'])

        imputer = SimpleImputer(strategy='median')
        X_train_i = imputer.fit_transform(X_train)
        X_test_i = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_i)
        X_test_s = scaler.transform(X_test_i)

        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        rf_preds = rf.predict(X_test_s)
        
        rf_acc = accuracy_score(y_test, rf_preds) * 100
        rf_bacc = balanced_accuracy_score(y_test, rf_preds) * 100
        rf_prec = precision_score(y_test, rf_preds, average='weighted', zero_division=0) * 100
        rf_rec = recall_score(y_test, rf_preds, average='weighted', zero_division=0) * 100
        rf_f1 = f1_score(y_test, rf_preds, average='weighted', zero_division=0) * 100

        # 2. XGBoost
        xgb = XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss')
        xgb.fit(X_train_s, y_train)
        xgb_preds = xgb.predict(X_test_s)
        
        xgb_acc = accuracy_score(y_test, xgb_preds) * 100
        xgb_bacc = balanced_accuracy_score(y_test, xgb_preds) * 100
        xgb_prec = precision_score(y_test, xgb_preds, average='weighted', zero_division=0) * 100
        xgb_rec = recall_score(y_test, xgb_preds, average='weighted', zero_division=0) * 100
        xgb_f1 = f1_score(y_test, xgb_preds, average='weighted', zero_division=0) * 100

        # Skriv ut detaljerade metrics direkt till terminalen
        print(f"\n--- RESULTAT FÖR INITIAL SOC: {int(soc*100)}% ---")
        print(f"  Testade på {len(real_sub)} verkliga cyklar.")
        print(f"  [Random Forest]")
        print(f"    - Accuracy         : {rf_acc:.2f}%")
        print(f"    - Balanced Accuracy: {rf_bacc:.2f}%")
        print(f"    - Precision (wtd)  : {rf_prec:.2f}%")
        print(f"    - Recall (wtd)     : {rf_rec:.2f}%")
        print(f"    - F1-score (wtd)   : {rf_f1:.2f}%")
        
        print(f"  [XGBoost]")
        print(f"    - Accuracy         : {xgb_acc:.2f}%")
        print(f"    - Balanced Accuracy: {xgb_bacc:.2f}%")
        print(f"    - Precision (wtd)  : {xgb_prec:.2f}%")
        print(f"    - Recall (wtd)     : {xgb_rec:.2f}%")
        print(f"    - F1-score (wtd)   : {xgb_f1:.2f}%")

        results["SOC"].append(soc * 100)
        results["RF_Balanced_Accuracy"].append(rf_bacc)
        results["XGB_Balanced_Accuracy"].append(xgb_bacc)

    return results


def plot_results(results):
    """Ritar grafer över balanced accuracy för respektive modell över olika Initial SOC."""
    if not results["SOC"]:
        print("Inga resultat att plotta.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(results["SOC"], results["RF_Balanced_Accuracy"], marker='o', linestyle='-', color='blue', linewidth=2, label='Random Forest')
    plt.plot(results["SOC"], results["XGB_Balanced_Accuracy"], marker='s', linestyle='--', color='orange', linewidth=2, label='XGBoost')

    plt.title("Balanced Accuracy vs. Initial SOC Level", fontsize=14)
    plt.xlabel("Initial SOC (%)", fontsize=12)
    plt.ylabel("Balanced Accuracy (%)", fontsize=12)
    
    plt.xlim(105, -2)  
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    plot_out_path = os.path.join(SCRIPT_DIR, "soc_accuracy_plot.png")
    plt.savefig(plot_out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[KLART] Jämförelsegraf har sparats till: {plot_out_path}")


def main():
    results = run_multi_soc_evaluation()
    plot_results(results)


if __name__ == "__main__":
    main()