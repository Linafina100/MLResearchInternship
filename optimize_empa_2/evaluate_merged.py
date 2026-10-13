"""
optimize_empa_2 merge draft: combines ideas from the three colleague
branches (hilda/main, ella, laswer5/optimize-runtime) that diverged
after the EMPA Initial_SOC sweep work. Base is hilda/main's
evaluate_empa_soc_sweep.py (depth-of-discharge real-data filter restored,
no hardcoded 2.5-3.0V target-zone restriction). Three changes on top:

1. TRAINING-ONLY bin coverage selection (ported from laswer5's
   optimize-runtime/features.py, which computes `>=20% coverage`
   against training data only). The shared root feature_engineering.py's
   own internal filter pools synthetic+real together by Chemistry,
   letting the real (test) side influence which bins get used for
   training -- a test-set-leakage-into-feature-selection issue. Fixed
   here by calling create_features_by_voltage_bins with
   min_chemistry_coverage=0.0 (disabling its internal filter) and
   re-implementing the >=20% check ourselves against the SYNTHETIC
   subset's own per-chemistry coverage only.

   (laswer5's polars/parquet/uv rewrite is NOT adopted wholesale --
   not installed in this environment, and this was the one
   self-contained algorithmic idea in it worth porting to the existing
   pandas pipeline everyone else here already runs.)

2. Per-SOC specialized models (ella's structural idea: a fresh model
   per real Initial_SOC bucket, rather than hilda/main's one model
   trained once on the full synthetic set and reused for every bucket).
   For buckets inside the synthetic generator's own Initial_SOC range
   (simulate_batteries.py samples SOC_RANGE_MIN=0.5 to SOC_RANGE_MAX=1.0
   -- a REST-then-continuous-discharge sample, not a truncation), the
   synthetic training set is narrowed to rows within
   SOC_MATCH_TOLERANCE of that bucket's target SOC, so each bucket's
   model is actually trained on SOC-appropriate data instead of the
   full pooled range.

   IMPORTANT CAVEAT, stated honestly rather than silently glossed over:
   the synthetic generator does not produce ANY data below Initial_SOC
   0.5. For every real bucket below 0.5 (0.4 down to 0.05 -- which is
   also where the sweep has historically collapsed), there is no
   SOC-matched synthetic data to narrow to, so those buckets fall back
   to the FULL synthetic set -- identical training data to a single
   pooled model, and therefore numerically IDENTICAL predictions to
   what hilda/main's one-model-for-everything approach would produce
   for that bucket. Per-SOC specialization as implemented here cannot,
   by itself, fix the low-SOC collapse. A real fix needs the synthetic
   generator extended to cover that range too, e.g. via the
   continuous-discharge-truncated approach already validated in
   optimize_empa/experiment/01_ratio_to_median_baseline/data_generation/
   (simulate one full discharge, then truncate it at each target SOC,
   the same way build_real_empa.py already truncates real cycles) --
   out of scope for this draft, flagged as the natural next step.

3. Ella's class_weight='balanced' on the Random Forest -- an orthogonal,
   safe improvement, adopted directly.

4. SOC_START_POINTS desync fix: hilda/main's evaluator requests 13
   points including 0.25, but build_real_empa.py (hers, and this
   file's build_real_empa_merged.py) only ever generates the standard
   12-point sweep -- that bucket was silently skipped every run. Dropped
   0.25 here to match the builder and the convention used everywhere
   else in this project.

Explicitly NOT adopted:
- Ella's capacity-fraction hybrid feature for SOC<=0.3 (replaces dV/dQ
  voltage bins with raw voltage resampled onto a 0-1 capacity-fraction
  grid) -- a materially different feature representation that would
  need its own validation before merging, not a drop-in compatible
  change with the rest of this pipeline's features.

Saves results to sweep_results_merged.json; reuses make_plot.py's
format (buckets keyed by SOC-as-string under "EMPA").

Usage: python3 optimize_empa_2/evaluate_merged.py
"""
import matplotlib
matplotlib.use("Agg")

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROJECT_DIR)  # for root feature_engineering.py

from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.environ.get("SYNTHETIC_RAW_CSV", os.path.join(PROJECT_DIR, "data", "raw", "advanced_synthetic_battery_data.csv"))
REAL_RAW_CSV = os.environ.get("REAL_RAW_CSV", os.path.join(PROJECT_DIR, "data", "raw", "real_empa_raw.csv"))
RAW_CSV = os.path.join(SCRIPT_DIR, "sim_and_real_raw_merged.csv")
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features_merged")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "sweep_results_merged.json")
GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']

SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]

# Below this, the synthetic generator (SOC_RANGE_MIN=0.5 in
# simulate_batteries.py) has no data at all -- every such bucket is
# extrapolation, flagged explicitly in the output (restores the
# EXTRAPOLATION_POINTS flag present in the original experiment 23/24
# lineage this fork is descended from, dropped somewhere along the way).
SYNTHETIC_SOC_FLOOR = 0.5
EXTRAPOLATION_POINTS = {s for s in SOC_START_POINTS if s < SYNTHETIC_SOC_FLOOR}

SOC_MATCH_TOLERANCE = float(os.environ.get("SOC_MATCH_TOLERANCE", 0.1))
MIN_MATCHED_TRAIN_ROWS = int(os.environ.get("MIN_MATCHED_TRAIN_ROWS", 50))
MIN_COVERAGE = 0.2


def build_combined_raw_csv():
    print(f"Loading synthetic data from '{SYNTHETIC_RAW_CSV}'...")
    sim_df = pd.read_csv(SYNTHETIC_RAW_CSV)
    sim_df["DataKind"] = "synthetic"
    print(f"  -> {len(sim_df)} rows")

    print(f"\nLoading real EMPA data from '{REAL_RAW_CSV}'...")
    real_df = pd.read_csv(REAL_RAW_CSV, low_memory=False)
    real_df["DataKind"] = "real"
    print(f"  -> {len(real_df)} rows, {real_df['Variation_ID'].nunique()} real discharge cycles")

    combined = pd.concat([sim_df, real_df], ignore_index=True)
    combined.to_csv(RAW_CSV, index=False)
    print(f"\nCombined raw data saved to '{RAW_CSV}': "
          f"{(combined['DataKind'] == 'synthetic').sum()} synthetic rows, "
          f"{(combined['DataKind'] == 'real').sum()} real rows")
    return combined


def extract_features_with_kind():
    if not os.path.exists(RAW_CSV):
        build_combined_raw_csv()
    else:
        print(f"Note: '{RAW_CSV}' already exists -- reusing it. Delete it to re-build from source.")

    print("\nExtracting RAW features (min_chemistry_coverage=0.0 -- no internal "
          "pooled synthetic+real filtering; coverage is re-checked below against "
          "the synthetic/training side only)...")
    features_df = create_features_by_voltage_bins(
        RAW_CSV, output_dir=FEATURES_DIR, min_chemistry_coverage=0.0, exclude_final_transition=True,
    )

    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df['Battery_ID'] = raw_df.groupby(GROUPBY_COLS).ngroup()
    id_cols = raw_df.drop_duplicates('Battery_ID').set_index('Battery_ID')[['DataKind', 'Initial_SOC']]
    features_df['DataKind'] = features_df['Battery_ID'].map(id_cols['DataKind'])
    features_df['Initial_SOC'] = features_df['Battery_ID'].map(id_cols['Initial_SOC'])

    all_v_cols = sorted(
        (c for c in features_df.columns if c.startswith('dV_dQ_V_')),
        key=lambda c: -float(c.split('_')[3]),
    )

    # Training-only (synthetic-only) >=20% per-chemistry coverage check --
    # the real/test side plays NO role in deciding which bins to keep.
    synth_only = features_df[features_df['DataKind'] == 'synthetic']
    coverage_by_chem = synth_only.groupby('Chemistry')[all_v_cols].apply(lambda g: g.notna().mean())
    bin_cols = [c for c in all_v_cols if (coverage_by_chem[c] >= MIN_COVERAGE).all()]
    dropped = [c for c in all_v_cols if c not in bin_cols]
    if dropped:
        print(f"Dropped {len(dropped)} bins with <{MIN_COVERAGE*100:.0f}% coverage in SYNTHETIC data "
              f"for at least one chemistry: {dropped}")

    print(f"\nSurviving bins (synthetic-only coverage check, {len(bin_cols)}): {bin_cols}")
    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries'))
    return features_df, bin_cols


def select_training_rows(synth_df, soc):
    """Per-SOC specialization: narrow synthetic training rows to those
    within SOC_MATCH_TOLERANCE of this bucket's target SOC, where the
    synthetic generator's own range (>=0.5) makes that possible. Falls
    back to the full synthetic set -- and flags is_soc_matched=False --
    whenever there isn't enough SOC-local data (always true below
    SYNTHETIC_SOC_FLOOR, since no synthetic data exists down there)."""
    if soc >= SYNTHETIC_SOC_FLOOR:
        matched = synth_df[(synth_df['Initial_SOC'] - soc).abs() <= SOC_MATCH_TOLERANCE]
        if len(matched) >= MIN_MATCHED_TRAIN_ROWS and matched['Chemistry'].nunique() == 2:
            return matched, True
    return synth_df, False


def fit_models(x_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced',
        ),
        "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def evaluate_bucket(soc, features_df, bin_cols):
    synth_df = features_df[features_df['DataKind'] == 'synthetic']
    real_bucket = features_df[(features_df.DataKind == 'real') & (features_df.Initial_SOC == soc)]
    if len(real_bucket) == 0:
        print(f"\n--- Initial_SOC={soc}: no surviving real batteries, skipped ---")
        return None

    train_df, is_soc_matched = select_training_rows(synth_df, soc)
    tag = " [SOC-MATCHED training]" if is_soc_matched else " [EXTRAPOLATION -- full synthetic range, no SOC match]"
    print(f"\n--- Initial_SOC={soc}{tag} ---")
    print(f"  training on {len(train_df)} synthetic batteries "
          f"(Initial_SOC {train_df['Initial_SOC'].min():.3f}-{train_df['Initial_SOC'].max():.3f})")

    X_train = train_df[bin_cols].replace([np.inf, -np.inf], np.nan)
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['Chemistry'])
    if len(set(y_train)) < 2:
        print(f"  SOC={soc}: synthetic side missing one chemistry after SOC-matching -- skipped.")
        return None

    lower, upper = X_train.quantile(0.01), X_train.quantile(0.99)
    X_train_c = X_train.clip(lower=lower, upper=upper, axis=1)
    imputer = SimpleImputer(strategy='median')
    X_train_i = imputer.fit_transform(X_train_c)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)

    models = fit_models(X_train_s, y_train)

    X_test = real_bucket[bin_cols].replace([np.inf, -np.inf], np.nan)
    y_test = le.transform(real_bucket['Chemistry'])
    X_test_c = X_test.clip(lower=lower, upper=upper, axis=1)
    X_test_i = imputer.transform(X_test_c)
    X_test_s = scaler.transform(X_test_i)

    out = {"n": int(len(real_bucket)), "is_soc_matched": is_soc_matched,
           "is_extrapolation": soc in EXTRAPOLATION_POINTS, "n_train": int(len(train_df))}
    for name, model in models.items():
        preds = model.predict(X_test_s)
        recalls = recall_score(y_test, preds, average=None, labels=[0, 1], zero_division=0)
        out[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "LFP_recall": float(recalls[0]),
            "NMC_recall": float(recalls[1]),
        }
        print(f"  {name:<15} balanced={out[name]['balanced_accuracy']*100:.2f}%  "
              f"LFP_recall={out[name]['LFP_recall']:.3f}  NMC_recall={out[name]['NMC_recall']:.3f}")
    return out


def main():
    features_df, bin_cols = extract_features_with_kind()

    if not bin_cols:
        print("\nNo surviving bins -- cannot train. Stopping here.")
        return

    print(f"\n{'=' * 70}\nPer-SOC specialized evaluation "
          f"(fresh model per bucket, SOC-matched synthetic training "
          f"where the generator's range allows it)\n{'=' * 70}")

    bucket_results = {}
    for soc in SOC_START_POINTS:
        result = evaluate_bucket(soc, features_df, bin_cols)
        if result is not None:
            bucket_results[str(soc)] = result

    print(f"\n{'=' * 70}\n=== SWEEP SUMMARY ===\n{'=' * 70}")
    print(f"Bins used ({len(bin_cols)}): {bin_cols}\n")
    header = (f"{'Initial_SOC':<12}{'n':<8}{'n_train':<9}{'matched':<9}{'RF balanced':<13}"
              f"{'RF LFP':<9}{'RF NMC':<9}{'XGB balanced':<14}{'XGB LFP':<9}{'XGB NMC':<9}")
    print(header)
    for soc in SOC_START_POINTS:
        s_str = str(soc)
        if s_str not in bucket_results:
            continue
        b = bucket_results[s_str]
        rf, xgb = b["Random Forest"], b["XGBoost"]
        extrap_tag = "*" if b["is_extrapolation"] else " "
        matched_tag = "yes" if b["is_soc_matched"] else "no"
        print(f"{soc}{extrap_tag:<11}{b['n']:<8}{b['n_train']:<9}{matched_tag:<9}"
              f"{rf['balanced_accuracy']*100:<13.2f}{rf['LFP_recall']:<9.3f}{rf['NMC_recall']:<9.3f}"
              f"{xgb['balanced_accuracy']*100:<14.2f}{xgb['LFP_recall']:<9.3f}{xgb['NMC_recall']:<9.3f}")
    print("\n* = extrapolation: below the synthetic generator's own Initial_SOC "
          f"range (>={SYNTHETIC_SOC_FLOOR}) -- no SOC-matched training data exists, "
          "falls back to the full synthetic set.")

    existing_results = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r") as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Note: Could not parse existing '{RESULTS_JSON}' ({e}). Rebuilding.")

    existing_results["EMPA"] = {"buckets": bucket_results}
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(existing_results, f, indent=2)
    print(f"\nSaved merged sweep results to '{RESULTS_JSON}'")


if __name__ == "__main__":
    main()
