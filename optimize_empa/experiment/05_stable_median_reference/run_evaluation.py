"""
optimize_empa_low_soc, attempt 05 (stable median reference): fixes a
specific problem found while investigating why attempt 04's XGBoost
works at Initial_SOC=0.05 (81.81%) but stays at chance at the LESS
extreme Initial_SOC=0.1, despite identical training data.

Root cause: ratio-to-median (attempt 04) normalizes each row by its
OWN median across its OWN observed bins -- but WHICH bins a row
observes depends on how deep its SOC truncation goes. At SOC=0.1, real
LFP rows have full coverage down to 3.2-3.1V; at SOC=0.05, LFP has ZERO
coverage at 3.3-3.2V AND 3.2-3.1V (confirmed directly: per-bin coverage
fractions differ sharply between the two buckets). So the row-median
denominator is computed over a DIFFERENT set of bins at each SOC point,
even though the underlying LFP-vs-NMC dV/dQ shape is qualitatively
similar at both (checked directly: LFP's ratio exceeds NMC's at the
same two deepest bins in both buckets). This shifts the resulting
ratio values in a SOC-dependent way that isn't about real
classification difficulty.

Fix: compute the row median over a FIXED, consistently-covered "core"
bin set (MEDIAN_REFERENCE_BINS, 2.6V-3.1V -- confirmed >=89% coverage
for both chemistries at the two most sparsely-covered buckets tested,
0.1 and 0.05) instead of whatever bins happen to be present per row.
Classification still uses the full bin_cols set (training-only
coverage, same as attempt 04) -- only the MEDIAN DENOMINATOR is fixed.
Verified before the full run: this makes the LFP-vs-NMC separation
pattern far more consistent across SOC=0.1 vs 0.05 (e.g. the 2.6-2.5V
bin's LFP-minus-NMC gap: 0.593 at both SOC points with the fix, vs.
0.573 and 0.354 -- a real SOC-dependent drift -- without it).

Reuses attempt 03's synthetic data (NEG_CAPACITY_FACTOR=0.45) and
attempt 01's real data unchanged, same as attempt 04. See this folder's
RESULTS.md for the full before/after result.

Also persists, per Initial_SOC bucket, whichever of RF/XGBoost scored
higher there (this is a per-SOC-specialized pipeline, not one global
model -- see outputs/models/README.md), plus everything needed to
reuse it: the fitted scaler/imputer/encoder, the clip bounds, and the
bin_cols / median-reference-bins used for the ratio-to-median
transform. Set SAVE_MODELS=0 to skip (results JSON is always written).

Usage: python3 optimize_empa/experiment/05_stable_median_reference/run_evaluation.py
"""
import matplotlib
matplotlib.use("Agg")

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

sys.path.insert(0, os.path.join(SCRIPT_DIR, "pipeline"))

from feature_engineering import create_features_by_voltage_bins

SYNTHETIC_RAW_CSV = os.environ.get("SYNTHETIC_RAW_CSV", os.path.join(PROJECT_DIR, "data", "optimize_empa_low_soc_03_synthetic", "raw", "advanced_synthetic_battery_data.csv"))
# Real data reused directly from attempt 01 -- real-data generation
# (filters, depth check, truncation) is unchanged by this attempt's fix,
# which only touches the synthetic generator. See this folder's
# RESULTS.md.
REAL_RAW_CSV = os.environ.get("REAL_RAW_CSV", os.path.join(PROJECT_DIR, "data", "optimize_empa_low_soc_01_real", "raw", "real_empa_raw.csv"))
FEATURES_DIR = os.path.join(SCRIPT_DIR, "outputs", "features")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "outputs", "sweep_results.json")
MODELS_DIR = os.path.join(SCRIPT_DIR, "outputs", "models")
SAVE_MODELS = os.environ.get("SAVE_MODELS", "1") not in ("0", "false", "False")

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
MIN_BINS_PER_ROW = int(os.environ.get("MIN_BINS_PER_ROW", 3))

# Fixed reference set for the ratio-to-median denominator -- see this
# file's docstring. Confirmed >=89% real-data coverage for both
# chemistries at Initial_SOC 0.1 and 0.05 (the two most sparsely
# covered buckets in the sweep), so should be at least as reliably
# covered at every less-truncated (higher Initial_SOC) bucket too.
MEDIAN_REFERENCE_BINS = [
    'dV_dQ_V_3.1_3.0', 'dV_dQ_V_3.0_2.9', 'dV_dQ_V_2.9_2.8',
    'dV_dQ_V_2.8_2.7', 'dV_dQ_V_2.7_2.6', 'dV_dQ_V_2.6_2.5',
]

# Real data at the full MAX_CYCLES_PER_CELL=600 scale (build_real_empa_merged.py's
# default) is ~12M raw rows / 1.7GB -- loading that, sorting it, and running
# the shared feature_engineering.py's vectorized diff/pivot pipeline on it as
# ONE DataFrame risks a repeat of an earlier out-of-memory crash on a
# different pipeline in this project. Processed in chunks of real
# Variation_IDs instead: each chunk is written to a small temp CSV, fed
# through the SAME shared create_features_by_voltage_bins() call used for
# everything else (no change to its logic, just smaller inputs), and only
# the resulting (small, one-row-per-battery) feature rows are kept --
# the large raw chunk is discarded before the next one is built. Synthetic
# data (498 batteries, never a memory concern) is extracted in one pass.
REAL_CHUNK_ROWS = int(os.environ.get("REAL_CHUNK_ROWS", 1_000_000))


def extract_synthetic_features():
    print(f"Extracting synthetic features from '{SYNTHETIC_RAW_CSV}' (single pass)...")
    synth_raw = pd.read_csv(SYNTHETIC_RAW_CSV)
    print(f"  -> {len(synth_raw)} rows")
    tmp_csv = os.path.join(SCRIPT_DIR, "_tmp_synth_only.csv")
    synth_raw.to_csv(tmp_csv, index=False)
    del synth_raw
    feats = create_features_by_voltage_bins(
        tmp_csv, output_dir=FEATURES_DIR, min_chemistry_coverage=0.0, exclude_final_transition=True,
    )
    os.remove(tmp_csv)
    feats["DataKind"] = "synthetic"
    return feats


def extract_real_features_chunked():
    print(f"\nLoading real EMPA data from '{REAL_RAW_CSV}'...")
    real_raw = pd.read_csv(REAL_RAW_CSV, low_memory=False)
    print(f"  -> {len(real_raw)} rows, {real_raw['Variation_ID'].nunique()} real discharge cycles")

    var_ids = real_raw["Variation_ID"].unique()
    rows_per_cycle = max(1, len(real_raw) // max(1, len(var_ids)))
    cycles_per_chunk = max(1, REAL_CHUNK_ROWS // rows_per_cycle)
    n_chunks = max(1, -(-len(var_ids) // cycles_per_chunk))  # ceil div
    chunks_of_varids = np.array_split(var_ids, n_chunks)

    print(f"  Processing in {n_chunks} chunks (~{cycles_per_chunk} cycles / "
          f"~{REAL_CHUNK_ROWS} rows each) to bound peak memory...")
    tmp_csv = os.path.join(SCRIPT_DIR, "_tmp_real_chunk.csv")
    all_feats = []
    for i, vids in enumerate(chunks_of_varids):
        chunk_df = real_raw[real_raw["Variation_ID"].isin(set(vids))]
        chunk_df.to_csv(tmp_csv, index=False)
        n_rows_chunk = len(chunk_df)
        del chunk_df
        feats = create_features_by_voltage_bins(
            tmp_csv, output_dir=FEATURES_DIR, min_chemistry_coverage=0.0, exclude_final_transition=True,
        )
        feats["DataKind"] = "real"
        all_feats.append(feats)
        print(f"  chunk {i+1}/{n_chunks}: {n_rows_chunk} raw rows -> {len(feats)} real batteries")
    os.remove(tmp_csv)
    del real_raw
    return pd.concat(all_feats, ignore_index=True)


def extract_features_with_kind():
    synth_feats = extract_synthetic_features()
    real_feats = extract_real_features_chunked()
    features_df = pd.concat([synth_feats, real_feats], ignore_index=True)

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


def apply_ratio_to_median(features_df, bin_cols, median_ref_bins):
    """Row-wise (per sample, synthetic and real alike). Each bin in
    bin_cols is divided by that row's own median computed over
    median_ref_bins ONLY (a fixed, consistently-covered subset -- see
    this file's docstring for why: using "whatever bins this row
    happens to observe" as attempt 04 did makes the denominator
    SOC-truncation-dependent, not just magnitude-independent). Rows
    with fewer than MIN_BINS_PER_ROW valid reference-bin values are
    dropped entirely (guards against a near-zero-variance median
    producing NaN/inf ratios)."""
    X = features_df[bin_cols].values.astype(float)
    X_ref = features_df[median_ref_bins].values.astype(float)
    n_valid = np.sum(~np.isnan(X_ref), axis=1)
    keep_mask = n_valid >= MIN_BINS_PER_ROW

    X_kept = X[keep_mask]
    X_ref_kept = X_ref[keep_mask]
    row_median = np.nanmedian(X_ref_kept, axis=1, keepdims=True)
    ratio = X_kept / row_median

    out = features_df.loc[keep_mask].copy()
    out[bin_cols] = ratio
    return out, int((~keep_mask).sum())


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


def save_bucket_model(soc, bin_cols, median_ref_bins, lower, upper, imputer, scaler, le, models, out):
    """Persists whichever of RF/XGBoost scored higher balanced_accuracy
    for this bucket, plus everything needed to run it on new data later:
    the fitted preprocessing (imputer/scaler/encoder), the per-bin clip
    bounds learned from this bucket's own training data, and the exact
    bin_cols / median_ref_bins this bucket's ratio-to-median transform
    used (both can differ bucket-to-bucket if coverage filtering drops
    different bins)."""
    winner_name = max(("Random Forest", "XGBoost"), key=lambda n: out[n]["balanced_accuracy"])
    bucket_dir = os.path.join(MODELS_DIR, f"soc_{soc}")
    os.makedirs(bucket_dir, exist_ok=True)

    joblib.dump(models[winner_name], os.path.join(bucket_dir, "classifier.joblib"))
    joblib.dump(scaler, os.path.join(bucket_dir, "scaler.joblib"))
    joblib.dump(imputer, os.path.join(bucket_dir, "imputer.joblib"))
    joblib.dump(le, os.path.join(bucket_dir, "label_encoder.joblib"))

    with open(os.path.join(bucket_dir, "feature_names.json"), "w") as f:
        json.dump({"bin_cols": list(bin_cols), "median_reference_bins": list(median_ref_bins)}, f, indent=2)
    with open(os.path.join(bucket_dir, "clip_bounds.json"), "w") as f:
        json.dump({"lower": lower.to_dict(), "upper": upper.to_dict()}, f, indent=2)
    with open(os.path.join(bucket_dir, "metadata.json"), "w") as f:
        json.dump({
            "initial_soc": soc, "model_name": winner_name,
            "is_soc_matched": out["is_soc_matched"], "is_extrapolation": out["is_extrapolation"],
            "n_train": out["n_train"], "n_test": out["n"],
            **{k: out[winner_name][k] for k in ("balanced_accuracy", "LFP_recall", "NMC_recall")},
        }, f, indent=2)
    print(f"  Saved winning model ({winner_name}, balanced={out[winner_name]['balanced_accuracy']*100:.2f}%) -> '{bucket_dir}'")


def evaluate_bucket(soc, features_df, bin_cols, median_ref_bins):
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

    if SAVE_MODELS:
        save_bucket_model(soc, bin_cols, median_ref_bins, lower, upper, imputer, scaler, le, models, out)

    return out


def main():
    features_df, bin_cols = extract_features_with_kind()

    if not bin_cols:
        print("\nNo surviving bins -- cannot train. Stopping here.")
        return

    median_ref_bins = [c for c in MEDIAN_REFERENCE_BINS if c in bin_cols]
    if len(median_ref_bins) < MIN_BINS_PER_ROW:
        print(f"\nWARNING: only {len(median_ref_bins)} of the intended "
              f"{len(MEDIAN_REFERENCE_BINS)} median-reference bins survived "
              f"coverage selection -- falling back to all bin_cols as the reference.")
        median_ref_bins = bin_cols
    features_df, n_dropped = apply_ratio_to_median(features_df, bin_cols, median_ref_bins)
    print(f"\nApplied ratio-to-median transform ({len(bin_cols)} bins -> row-wise ratios, "
          f"median computed over the fixed {len(median_ref_bins)}-bin reference set "
          f"{median_ref_bins}). Dropped {n_dropped} rows with < {MIN_BINS_PER_ROW} valid "
          f"reference-bin values.")
    print(features_df.groupby(['DataKind', 'Chemistry']).size().rename('n_batteries (post ratio-to-median)'))

    print(f"\n{'=' * 70}\nPer-SOC specialized evaluation "
          f"(fresh model per bucket, SOC-matched synthetic training "
          f"where the generator's range allows it)\n{'=' * 70}")

    bucket_results = {}
    for soc in SOC_START_POINTS:
        result = evaluate_bucket(soc, features_df, bin_cols, median_ref_bins)
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
