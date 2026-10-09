"""Single-pass SPM discharge simulations with deterministic resumable jobs."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY', 'true')

import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import polars as pl
import pybamm

from workflow import DATA, SYNTH_RAW, atomic_json, atomic_parquet, export_csv, fingerprint, path_setting


def setting(name, default):
    return float(os.environ.get(name, default))


def make_jobs():
    rng = random.Random(42)
    count = int(os.environ.get('VARIATIONS_PER_SIZE', 83))
    if count < 1:
        raise ValueError('VARIATIONS_PER_SIZE must be positive')
    ranges = {name: (setting(f'{name}_MIN', low), setting(f'{name}_MAX', high)) for name, low, high in
              [('SOC_RANGE', 0.5, 1.0), ('SOH', 0.8, 1.0), ('C_RATE', 0.1, 0.2)]}
    if any(low <= 0 or low > high for low, high in ranges.values()) or ranges['SOC_RANGE'][1] > 1:
        raise ValueError('Invalid SOC, SOH, or C-rate range')
    jobs = []
    for size_idx, target in enumerate([1.2, 2.0, 3.5]):
        for i in range(count):
            soc, soh, c_rate = [rng.uniform(*ranges[name]) for name in ('SOC_RANGE', 'SOH', 'C_RATE')]
            ambient = rng.uniform(15, 35)
            nmc_set = rng.choice(['Chen2020', 'OKane2022'])
            resistance = {chem: min(1., max(.3, soh * rng.uniform(.8, 1.2) * (1 + .02 * (ambient - 25))))
                          for chem in ('LFP', 'NMC')}
            for chem in ('LFP', 'NMC'):
                jobs.append({'chem': chem, 'target': target, 'soc': soc, 'soh': soh, 'c_rate': c_rate,
                             'ambient': ambient, 'variation': size_idx * count + i,
                             'parameter_set': nmc_set if chem == 'NMC' else 'Prada2013',
                             'resistance': resistance[chem], 'cutoff': setting(f'{chem}_LOWER_CUTOFF', 1.5),
                             'diffusivity': setting('DIFFUSIVITY_FACTOR', 10), 'ocp_rate': setting('LFP_OCP_RATE_CONSTANT', -3)})
    for job, seed in zip(jobs, np.random.SeedSequence(42).spawn(len(jobs))):
        job['seed'] = int(seed.generate_state(1)[0])
    return jobs


@lru_cache(maxsize=3)
def base_parameters(name):
    return pybamm.ParameterValues(name)


def solve_job(task):
    job, folder = task
    chem = job['chem']
    job_id = f'{chem}-{job["variation"]:05d}'
    part = folder / f'{job_id}.parquet'
    marker = folder / f'{job_id}.json'
    try:
        param = base_parameters(job['parameter_set']).copy()
        base = base_parameters('Prada2013' if chem == 'LFP' else 'Chen2020')
        mult = job['target'] / base['Nominal cell capacity [A.h]']
        if chem == 'LFP':
            rate = job['ocp_rate']
            param['Positive electrode OCP [V]'] = lambda sto: 3.4077 - .020269 * sto + .5 * np.exp(-150 * sto) - .9 * np.exp(rate * (1 - sto))
        else:
            original = param['Positive particle diffusivity [m2.s-1]']
            factor = job['diffusivity']
            param['Positive particle diffusivity [m2.s-1]'] = ((lambda sto, temp: original(sto, temp) / factor)
                                                              if callable(original) else original / factor)
        for electrode in ('negative', 'positive'):
            title = electrode.title()
            param[f'{title} electrode thickness [m]'] *= mult
            param[f'Maximum concentration in {electrode} electrode [mol.m-3]'] *= job['soh']
            param[f'Initial concentration in {electrode} electrode [mol.m-3]'] *= job['soh']
            param[f'{title} electrode conductivity [S.m-1]'] *= job['resistance']
        current = job['c_rate'] * job['target']
        experiment = pybamm.Experiment([f'Discharge at {current:.4f} A until {job["cutoff"]} V'])
        n_points = max(100, round(3000 * job['soc']))
        spacing = (3600 * job['soc'] / job['c_rate']) / (n_points - 1)
        horizon = experiment.steps[0].duration
        times = np.arange(0, horizon, spacing)
        times = np.r_[times, horizon]
        solver = pybamm.IDAKLUSolver(options={'num_threads': 1})
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), parameter_values=param, experiment=experiment, solver=solver)
        sol = sim.solve(initial_soc=job['soc'], t_interp=times)
        t = sol['Time [s]'].entries
        v = sol['Terminal voltage [V]'].entries
        q = sol['Discharge capacity [A.h]'].entries
        if len(t) < 15 or not np.isfinite(np.column_stack([t, v, q])).all():
            raise ValueError('Simulation did not produce a usable finite trace')
        if sol.termination == 'final time':
            raise ValueError('Experiment horizon reached before the voltage cutoff')
        noise = np.random.default_rng(job['seed']).normal(0, .001, len(t))
        metadata = {'Chemistry': chem, 'DataKind': 'synthetic', 'Cell_ID': job_id,
                    'Variation_ID': str(job['variation']), 'Target_Capacity_Ah': job['target'],
                    'C_Rate': round(job['c_rate'], 4), 'Size_Multiplier': mult, 'SOH': round(job['soh'], 3),
                    'Initial_SOC': round(job['soc'], 3), 'Ambient_Temperature_C': round(job['ambient'], 2),
                    'Resistance_Factor': round(job['resistance'], 4), 'Base_Parameter_Set': job['parameter_set'],
                    'V_min [V]': job['cutoff']}
        frame = pl.DataFrame({'Time [s]': t, 'Voltage [V]': v + noise, 'Capacity [A.h]': q}).with_columns(
            [pl.lit(value).alias(name) for name, value in metadata.items()])
        atomic_parquet(frame, part)
        result = {'job': job_id, 'rows': len(t), 'part': part.name, 'termination': sol.termination, 'status': 'complete'}
        atomic_json(marker, result)
        return result
    except Exception as exc:
        return {'job': job_id, 'rows': 0, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}


def plot_dataset(parts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    labeled = set()
    for part in parts:
        frame = pl.read_parquet(part)
        chem = frame['Chemistry'][0]
        ax.plot(frame['Time [s]'].to_numpy() / 3600, frame['Voltage [V]'].to_numpy(),
                color='#1f77b4' if chem == 'LFP' else '#ff7f0e', alpha=.15, linewidth=.8,
                label=chem if chem not in labeled else None)
        labeled.add(chem)
    ax.set(xlabel='Time [Hours]', ylabel='Voltage [V]', title='Simulated continuous discharge profiles')
    ax.legend()
    ax.grid(True)
    target = path_setting('DISCHARGE_PLOT_PNG', DATA / 'plots' / 'continuous_discharge_plot.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    jobs = make_jobs()
    key = fingerprint(settings=jobs)
    folder = SYNTH_RAW / 'parts' / key
    folder.mkdir(parents=True, exist_ok=True)
    pending, results = [], []
    for job in jobs:
        stem = f'{job["chem"]}-{job["variation"]:05d}'
        marker = folder / f'{stem}.json'
        if os.environ.get('USE_CACHE', '1') != '0' and marker.exists() and (folder / f'{stem}.parquet').exists():
            results.append(json.loads(marker.read_text()) | {'cached': True})
        else:
            pending.append((job, folder))
    workers = int(os.environ.get('SIM_WORKERS', min(4, os.cpu_count() or 1)))
    if workers < 1:
        raise ValueError('SIM_WORKERS must be positive')
    if workers == 1:
        for task in pending:
            result = solve_job(task)
            results.append(result)
            print(json.dumps(result), flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(solve_job, pending):
                results.append(result)
                print(json.dumps(result), flush=True)
    results.sort(key=lambda r: r['job'])
    failures = [r for r in results if r['status'] != 'complete']
    complete = [r for r in results if r['status'] == 'complete']
    manifest = {'parts': [str((folder / r['part']).relative_to(SYNTH_RAW)) for r in complete],
                'rows': sum(r['rows'] for r in complete), 'variants': len(complete), 'attempts': len(jobs),
                'failures': failures, 'jobs': results, 'workers': workers}
    atomic_json(SYNTH_RAW / 'manifest.json', manifest)
    failure_path = path_setting('FAILURE_LOG_CSV', DATA / 'failures' / 'simulation_failures.csv')
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(failures, schema={'job': pl.String, 'rows': pl.Int64, 'status': pl.String, 'error': pl.String}).write_csv(failure_path)
    if not complete:
        raise RuntimeError('Every simulation failed; inspect the failure log')
    if os.environ.get('OUTPUT_DATA_CSV') or os.environ.get('EXPORT_CSV') == '1':
        export_csv(SYNTH_RAW, path_setting('OUTPUT_DATA_CSV', SYNTH_RAW.with_suffix('.csv')))
    plot_dataset([SYNTH_RAW / p for p in manifest['parts']])
    print(f'{len(complete)}/{len(jobs)} traces completed; {len(failures)} failures; {len(results) - len(pending)} cached', flush=True)


if __name__ == '__main__':
    main()
