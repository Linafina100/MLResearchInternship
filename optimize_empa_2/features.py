"""Vectorized trace-to-voltage-bin features."""
import numpy as np
import polars as pl

from workflow import FEATURES, atomic_parquet, data_files, fingerprint, read_table

KEYS = ['DataKind', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
META = ['Cell_ID', 'Cycle', 'Target_Capacity_Ah', 'C_Rate', 'Ambient_Temperature_C',
        'Resistance_Factor', 'Base_Parameter_Set', 'Real_BOL_Capacity_mAh']
BINS = [f'dV_dQ_V_{high / 10:.1f}_{(high - 1) / 10:.1f}' for high in range(43, 19, -1)]


def extract_table(frame, kind):
    frame = frame.with_columns(pl.lit(kind).alias('DataKind'), pl.col('Variation_ID').cast(pl.String))
    frame = frame.drop_nulls(KEYS).sort(KEYS + ['Time [s]'], maintain_order=True)
    if frame.is_empty():
        return pl.DataFrame()
    codes = frame.select(pl.struct(KEYS).rle_id()).to_series().to_numpy().astype(np.int64)
    starts = np.r_[True, codes[1:] != codes[:-1]]
    n_groups = int(codes[-1]) + 1
    sizes = np.bincount(codes, minlength=n_groups)
    voltage = frame['Voltage [V]'].to_numpy()
    capacity = frame['Capacity [A.h]'].to_numpy()
    dv = np.diff(voltage, prepend=np.nan)
    dq = np.diff(capacity, prepend=np.nan)
    valid = (~starts) & (dq > 1e-5) & (sizes[codes] >= 15)
    idx = np.flatnonzero(valid)
    valid_counts = np.bincount(codes[idx], minlength=n_groups)
    last = np.full(n_groups, -1, dtype=np.int64)
    np.maximum.at(last, codes[idx], idx)
    valid[last[valid_counts > 1]] = False
    valid &= np.isfinite(voltage) & np.isfinite(dv) & np.isfinite(dq)
    idx = np.flatnonzero(valid)
    highs = np.clip(np.ceil(np.round(voltage[idx], 4) * 10), 20, 43).astype(np.int64)
    bin_codes = codes[idx] * 24 + (43 - highs)
    sums = np.bincount(bin_codes, weights=dv[idx] / dq[idx], minlength=n_groups * 24)
    counts = np.bincount(bin_codes, minlength=n_groups * 24)
    values = np.full(n_groups * 24, np.nan)
    np.divide(sums, counts, out=values, where=counts > 0)
    values = values.reshape(n_groups, 24)
    metadata = frame.filter(pl.Series(starts)).select(KEYS + [c for c in META if c in frame.columns])
    metadata = metadata.with_columns(pl.struct(KEYS).struct.json_encode().alias('Battery_ID'))
    return metadata.hstack(pl.DataFrame(values, schema=BINS, orient='row', nan_to_null=True)).filter(
        pl.Series(counts.reshape(n_groups, 24).sum(axis=1) > 0)
    )


def extract_features(source, kind, cache=True, cache_dir=FEATURES):
    if isinstance(source, pl.DataFrame):
        return extract_table(source, kind)
    files = data_files(source)
    if not files:
        raise ValueError(f'No raw traces in {source}')
    key = fingerprint(files, {'kind': kind, 'bins': BINS, 'exclude_final_transition': True})
    target = cache_dir / f'{kind}-{key}.parquet'
    if cache and target.exists():
        print(f'Feature cache: {target}', flush=True)
        return pl.read_parquet(target)
    frames = [extract_table(read_table(p), kind) for p in files]
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        raise ValueError(f'No valid feature rows in {source}')
    result = pl.concat(frames, how='diagonal_relaxed')
    if result['Battery_ID'].n_unique() != result.height:
        raise ValueError('A trace spans multiple input files; store complete traces in each part')
    atomic_parquet(result, target)
    return result


def select_bins(training, min_coverage=0.2):
    if set(training['Chemistry'].unique()) != {'LFP', 'NMC'}:
        raise ValueError('Training requires both LFP and NMC traces')
    coverage = training.group_by('Chemistry').agg(
        [pl.col(c).is_finite().fill_null(False).mean().alias(c) for c in BINS]
    )
    return [c for c in BINS if coverage[c].min() >= min_coverage and coverage[c].max() > 0]
