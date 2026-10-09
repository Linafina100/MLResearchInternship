"""Shared training, preprocessing, and pooled/SOC evaluation."""
import os

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
# from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import extract_features, select_bins
from workflow import HERE, REAL_RAW, SOC_START_POINTS, SYNTH_RAW, atomic_json, path_setting


def matrix(frame, columns):
    values = frame.select(columns).to_numpy().copy()
    values[~np.isfinite(values)] = np.nan
    return values


def labels(frame):
    values = frame['Chemistry'].to_numpy()
    if not set(values) <= {'LFP', 'NMC'}:
        raise ValueError('Unknown chemistry in feature table')
    return (values == 'NMC').astype(np.int64)


def evaluate_features(synthetic, real, sweep=True):
    columns = select_bins(synthetic)
    if not columns:
        raise ValueError('No bins meet the training-only mutual-coverage threshold')
    if real.is_empty():
        raise ValueError('No real feature rows')
    x_train, x_test = matrix(synthetic, columns), matrix(real, columns)
    lower, upper = np.nanquantile(x_train, [.01, .99], axis=0)
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(np.clip(x_train, lower, upper))
    x_test = imputer.transform(np.clip(x_test, lower, upper))
    # Scaling behövs inte för "scale-invariant"-modeller som Random Forest och tree-based XGBoost.
    # Var försiktiga dock om ni skulle få för er att testa andra modeller. 
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    y_train, y_test = labels(synthetic), labels(real)
    workers = int(os.environ.get('ML_WORKERS', min(4, os.cpu_count() or 1)))
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=workers),
        'XGBoost': XGBClassifier(n_estimators=150, learning_rate=.08, max_depth=4, subsample=.8,
                                 colsample_bytree=.8, eval_metric='logloss', random_state=42, n_jobs=workers),
    }
    masks = {'pooled': np.ones(real.height, dtype=bool)}
    if sweep:
        soc = real['Initial_SOC'].to_numpy()
        masks.update({str(s): soc == s for s in SOC_START_POINTS if np.any(soc == s)})
    results = {name: {} for name in masks}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        for bucket, mask in masks.items():
            truth, preds = y_test[mask], predictions[mask]
            recall = recall_score(truth, preds, labels=[0, 1], average=None, zero_division=0)
            result = {'n': int(mask.sum()), 'n_lfp': int((truth == 0).sum()), 'n_nmc': int((truth == 1).sum()),
                      'accuracy': float(accuracy_score(truth, preds)),
                      'balanced_accuracy': float(balanced_accuracy_score(truth, preds)),
                      'LFP_recall': float(recall[0]), 'NMC_recall': float(recall[1])}
            results[bucket][name] = result
            print(f'{bucket:>6} {name:14}: n={result["n"]}, balanced={result["balanced_accuracy"]:.4f}', flush=True)
    return {'bins': columns, 'training_traces': synthetic.height, 'real_traces': real.height,
            'pooled': results.pop('pooled'), 'buckets': results}


def main(sweep=True):
    synthetic_source = path_setting('SYNTHETIC_RAW_CSV', SYNTH_RAW)
    real_source = path_setting('REAL_RAW_CSV', REAL_RAW)
    cache = os.environ.get('USE_CACHE', '1') != '0'
    synthetic = extract_features(synthetic_source, 'synthetic', cache=cache)
    real = extract_features(real_source, 'real', cache=cache)
    result = evaluate_features(synthetic, real, sweep=sweep)
    target = path_setting('RESULTS_JSON', HERE / ('sweep_results.json' if sweep else 'evaluation_results.json'))
    atomic_json(target, {'EMPA': result})
    print(f'Results: {target}', flush=True)
    return result
