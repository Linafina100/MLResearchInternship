"""Shared paths, cache fingerprints, and atomic dataset publication."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOC_START_POINTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
PACKAGES = ('numpy', 'polars', 'pybamm', 'scikit-learn', 'xgboost')


def path_setting(name, default):
    path = Path(os.environ.get(name, str(default))).expanduser()
    return path if path.is_absolute() else PROJECT / path


DATA = path_setting('DATA_DIR', PROJECT / 'data')
EMPA = path_setting('EMPA_DIR', DATA / 'Dataset-rocrate')
REAL_RAW = path_setting('REAL_RAW_PARQUET', DATA / 'raw' / 'real_empa_raw')
SYNTH_RAW = path_setting('OUTPUT_DATA_PARQUET', DATA / 'raw' / 'advanced_synthetic_battery_data')
FEATURES = path_setting('FEATURES_DIR', DATA / 'features' / 'optimize_empa_2')


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(files=(), settings=None):
    payload = {
        'inputs': [(str(Path(p).resolve()), digest_file(p)) for p in sorted(map(Path, files))],
        'settings': settings,
        'source': {p.name: digest_file(p) for p in sorted(HERE.glob('*.py'))},
        'versions': versions(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    frame.write_parquet(tmp, compression='zstd')
    tmp.replace(path)


def data_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    manifest = json.loads((path / 'manifest.json').read_text())
    return [path / name for name in manifest['parts']]


def read_table(path):
    path = Path(path)
    if path.suffix == '.csv':
        return pl.read_csv(path, infer_schema_length=10000, schema_overrides={'Variation_ID': pl.String})
    return pl.read_parquet(path)


def export_csv(dataset, destination):
    files = data_files(dataset)
    if not files:
        raise ValueError('Dataset contains no traces')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + '.tmp')
    pl.scan_parquet(files).sink_csv(tmp)
    tmp.replace(destination)
