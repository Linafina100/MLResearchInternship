"""
Experiment 32: concatenates simulate_lfp_negative_electrode_balance.py's
LFP-only output (negative-electrode capacity factor=0.7, see this
folder's RESULTS.md for how that value was found and validated) with the
EXISTING, UNMODIFIED NMC rows from
data/26_soh_range_continuous_discharge_truncated_v1/ -- NMC was never
touched by this fix, so it's reused directly rather than resimulated.

Usage: python3 experiments/32_fixed_voltage_cutoff_soc_sweep/combine_lfp_fix_with_baseline_nmc.py
"""
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

BASELINE_CSV = os.path.join(PROJECT_DIR, "data", "26_soh_range_continuous_discharge_truncated_v1", "raw", "advanced_synthetic_battery_data.csv")
LFP_FIX_CSV = os.path.join(PROJECT_DIR, "data", "32_lfp_negative_electrode_balance", "raw", "lfp_only_balance_fix.csv")
OUT_CSV = os.path.join(PROJECT_DIR, "data", "32_synthetic_lfp_balance_fix_combined", "raw", "advanced_synthetic_battery_data.csv")


def main():
    print(f"Loading baseline (for NMC rows, unchanged) from '{BASELINE_CSV}'...")
    baseline = pd.read_csv(BASELINE_CSV, low_memory=False)
    nmc = baseline[baseline["Chemistry"] == "NMC"].copy()
    print(f"  -> {len(nmc)} NMC rows, {nmc['Variation_ID'].str.split('_soc_').str[0].nunique()} base configs")

    print(f"Loading LFP balance-fix rows from '{LFP_FIX_CSV}'...")
    lfp_fixed = pd.read_csv(LFP_FIX_CSV, low_memory=False)
    print(f"  -> {len(lfp_fixed)} LFP rows, {lfp_fixed['Variation_ID'].str.split('_soc_').str[0].nunique()} base configs")

    common_cols = [c for c in nmc.columns if c in lfp_fixed.columns]
    combined = pd.concat([nmc[common_cols], lfp_fixed[common_cols]], ignore_index=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\n-> {OUT_CSV}")
    print(f"   {len(combined)} rows")
    print(combined.groupby("Chemistry")["Variation_ID"].apply(lambda s: s.str.split("_soc_").str[0].nunique()).rename("n_base_configs"))


if __name__ == "__main__":
    main()
