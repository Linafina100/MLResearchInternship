"""Build EMPA SOC variants as a resumable partitioned Parquet dataset."""
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from workflow import EMPA, REAL_RAW, SOC_START_POINTS, atomic_json, atomic_parquet, export_csv, fingerprint, path_setting

SOH_MIN = float(os.environ.get('SOH_MIN', 0.8))
SOH_MAX = float(os.environ.get('SOH_MAX', 1.0))
C_RATE_MIN = float(os.environ.get('C_RATE_MIN', 0.0))
C_RATE_MAX = float(os.environ.get('C_RATE_MAX', 999.0))
CHEMISTRIES = {'LithiumIronPhosphateOxide': 'LFP', 'LithiumNickelCobaltManganeseOxide': 'NMC'}
COLS = ['cycle_dimensionless', 'test_time_millisecond', 'current_ampere', 'voltage_volt', 'ambient_temperature_celsius']


def get_chemistry(cell_dir, cell_id=None):
    cell_dir = Path(cell_dir)
    cell_id = cell_id or cell_dir.name
    meta = json.loads((cell_dir / f'{cell_id}.metadata.json').read_text())
    try:
        active = meta['@graph'][0]['hasTestObject']['hasPositiveElectrode']['hasCoating']['hasActiveMaterial']['rdfs:comment']
        return CHEMISTRIES.get(active)
    except (KeyError, IndexError, TypeError):
        return None


def selected_cells(directory=EMPA):
    wanted = set(filter(None, os.environ.get('EMPA_CELL_IDS', '').split(',')))
    return [p for p in sorted(Path(directory).glob('empa__ccid*'))
            if p.is_dir() and (not wanted or p.name in wanted)]


def capacity(time_ms, current):
    seconds = (time_ms - time_ms[0]) / 1000.0
    q = np.r_[0.0, np.cumsum(np.diff(seconds) * (np.abs(current[:-1]) + np.abs(current[1:])) / 7200)]
    return seconds, q


def truncate_at_soc(t, v, q, soc):
    threshold = (1 - soc) * q[-1]
    if threshold > 0:
        start = np.searchsorted(q, threshold, side='right')
        if len(q) - start < 4:
            return None
        t = np.r_[np.interp(threshold, q, t), t[start:]]
        v = np.r_[np.interp(threshold, q, v), v[start:]]
        q = np.r_[threshold, q[start:]]
        t, q = t - t[0], q - q[0]
    if len(t) < 5 or v.min() >= 3.0:
        return None
    return t, v, q


def process_cell(cell_dir, output=REAL_RAW, cache=True):
    cell_dir, output = Path(cell_dir), Path(output)
    cell_id = cell_dir.name
    chemistry = get_chemistry(cell_dir)
    if chemistry is None:
        return {'cell': cell_id, 'parts': [], 'rows': 0, 'variants': 0, 'skipped': 'chemistry'}
    source = cell_dir / f'{cell_id}.bdf.parquet'
    key = fingerprint([source, cell_dir / f'{cell_id}.metadata.json'],
                      {'soh': [SOH_MIN, SOH_MAX], 'c_rate': [C_RATE_MIN, C_RATE_MAX], 'soc': SOC_START_POINTS})
    folder = output / 'parts' / f'{cell_id}-{key}'
    marker = folder / 'complete.json'
    if cache and marker.exists():
        result = json.loads(marker.read_text())
        if all((output / part).exists() for part in result['parts']):
            return result | {'cached': True}
    frame = (pl.scan_parquet(source).select(COLS)
             .filter(pl.col('cycle_dimensionless').is_not_null() & (pl.col('current_ampere') < -5e-7))
             .sort(['cycle_dimensionless', 'test_time_millisecond'], maintain_order=True).collect())
    arr = frame.select(pl.all().cast(pl.Float64)).to_numpy()
    segments = []
    if len(arr):
        boundaries = np.r_[0, np.flatnonzero(np.diff(arr[:, 0]) != 0) + 1, len(arr)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            seg = arr[start:end]
            if len(seg) >= 15 and (seg[-1, 1] - seg[0, 1]) <= 86_400_000 and np.ptp(seg[:, 3]) >= 0.3:
                segments.append((int(seg[0, 0]), start, end))
    early = [s for s in segments if s[0] < 10] or segments[:10]
    bol = max((capacity(arr[a:b, 1], arr[a:b, 2])[1][-1] for _, a, b in early), default=0)
    result = {'cell': cell_id, 'chemistry': chemistry, 'parts': [], 'rows': 0, 'variants': 0,
              'cycles_scanned': len(segments), 'cycles_matched': 0, 'bol_ah': float(bol)}
    chunks = []

    def flush():
        if not chunks:
            return
        columns = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
        part = folder / f'{len(result["parts"]):05d}.parquet'
        atomic_parquet(pl.DataFrame(columns), part)
        result['parts'].append(str(part.relative_to(output)))
        chunks.clear()

    if bol > 0:
        for cycle, start, end in segments:
            seg = arr[start:end]
            t, q = capacity(seg[:, 1], seg[:, 2])
            soh, c_rate = q[-1] / bol, np.abs(seg[:, 2]).mean() / bol
            if not (SOH_MIN <= soh <= SOH_MAX and C_RATE_MIN <= c_rate <= C_RATE_MAX):
                continue
            result['cycles_matched'] += 1
            for soc in SOC_START_POINTS:
                truncated = truncate_at_soc(t, seg[:, 3], q, soc)
                if truncated is None:
                    continue
                tt, vv, qq = truncated
                qq = qq / bol * 2.0
                grid = np.linspace(qq[0], qq[-1], 80)
                metadata = {'Chemistry': chemistry, 'Variation_ID': f'{cell_id}_cycle_{cycle}_soc_{soc}',
                            'Cell_ID': cell_id, 'Cycle': cycle, 'DataKind': 'real',
                            'Target_Capacity_Ah': 2.0, 'Real_BOL_Capacity_mAh': bol * 1000,
                            'Size_Multiplier': 0.0, 'SOH': round(float(soh), 4), 'Initial_SOC': soc,
                            'Ambient_Temperature_C': float(np.nanmean(seg[:, 4]))}
                chunk = {name: np.full(80, value) for name, value in metadata.items()}
                chunk.update({'Time [s]': np.interp(grid, qq, tt), 'Voltage [V]': np.interp(grid, qq, vv), 'Capacity [A.h]': grid})
                chunks.append(chunk)
                result['variants'] += 1
                result['rows'] += 80
                if len(chunks) >= 512:
                    flush()
    flush()
    atomic_json(marker, result)
    return result


def main():
    cells = selected_cells()
    if not cells:
        raise FileNotFoundError(f'No EMPA cells in {EMPA}')
    results = []
    for i, cell in enumerate(cells, 1):
        result = process_cell(cell, cache=os.environ.get('USE_CACHE', '1') != '0')
        results.append(result)
        print(f'[{i}/{len(cells)}] {cell.name}: {result["variants"]} variants, cached={result.get("cached", False)}', flush=True)
    manifest = {'parts': [p for result in results for p in result['parts']], 'cells': results,
                'rows': sum(r['rows'] for r in results), 'variants': sum(r['variants'] for r in results)}
    if not manifest['parts']:
        raise ValueError('No matching EMPA traces')
    atomic_json(REAL_RAW / 'manifest.json', manifest)
    if os.environ.get('REAL_RAW_CSV') or os.environ.get('EXPORT_CSV') == '1':
        export_csv(REAL_RAW, path_setting('REAL_RAW_CSV', REAL_RAW.with_suffix('.csv')))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('parts', 'cells')}), flush=True)


if __name__ == '__main__':
    main()
