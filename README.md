# MLResearchInternship

Classifying lithium-ion battery chemistry (LFP vs. NMC) from discharge
voltage curves, using PyBaMM-simulated cells (and, increasingly, real
experimental data) and a dV/dQ voltage-bin feature set fed into Random
Forest / XGBoost classifiers. Originally built around a GITT pulse-discharge
protocol; the main pipeline now models a continuous constant-current
discharge instead, closer to how the intended real-world test equipment
(Stena) actually runs cells.

## Repository layout

```
simulate_batteries.py      # Simulates LFP/NMC discharge curves (PyBaMM), the "main" pipeline
feature_engineering.py     # Extracts dV/dQ voltage-bin features from simulated/raw discharge data
ml_pipeline.py             # Trains + evaluates Random Forest / XGBoost on those features
models/                    # Tracked, trained model artifacts (scaler, imputer, encoder, feature list)
data/                      # Gitignored: all simulation/feature/plot output, regenerated per run
experiments/               # Numbered, self-contained experiments -- see below
```

### Root pipeline

`simulate_batteries.py` runs a continuous discharge (fixed 0.6C by default)
for randomized LFP/NMC cell variations (SOC, SOH, ambient temperature,
resistance) down to each chemistry's voltage cutoff, writing raw
voltage/capacity traces plus a discharge-curve plot. `feature_engineering.py`
bins the resulting dV/dQ signal into 0.1V-wide voltage bins and drops any
bin only one chemistry ever reaches (a leakage fix -- see experiment 01).
`ml_pipeline.py` trains/evaluates classifiers on those features and can
optionally persist the trained model + preprocessing artifacts to
`models/`. All three read their key parameters (SOC range, output paths,
voltage cutoffs) from environment variables so sweep scripts can drive many
systematic runs without duplicating this setup -- see any `experiments/*/sweep_*.py`
for the pattern.

Run directly with no arguments for a single default run:
```
python3 simulate_batteries.py
python3 feature_engineering.py
python3 ml_pipeline.py   # or import run_ml_pipeline from another script
```

### Dependencies

No lockfile is currently checked in. The pipeline imports: `pybamm`,
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`,
`joblib`.

## Experiments

Each `experiments/NN_<name>/` folder is a self-contained variation on the
main pipeline -- usually its own `RESULTS.md` write-up, plus one or more
scripts that either fork the root pipeline's simulation/feature logic or
reuse it directly and just sweep a parameter. Numbering collides
occasionally when multiple people add a folder independently; always check
`ls experiments/` on a fresh pull before assigning a new number, and prefer
renumbering whichever folder nothing else references by path over one that
other code imports.

| # | Folder | What it does |
|---|---|---|
| 01 | `01_leakage_fix_20_percent_coverage` | Original SOC-availability sweep (pulse protocol); introduced the `>=20%` mutual-coverage leakage fix that drops voltage bins only one chemistry ever reaches. |
| 02 | `02_avoid_nan_soc_binning` | Fixes a NaN-handling bug in SOC binning from experiment 01. |
| 03 | `03_continuous_discharge_soc_sweep` | Replaces the GITT pulse protocol with continuous constant-current discharge (fixed 0.6C) and re-runs the same 4-interval SOC sweep; this became the main pipeline. |
| 04 | `04_pulse_variable_discharge_soc_sweep` | Pulse protocol with a per-battery randomized discharge rate; found this collapses signal almost entirely due to a sampling-resolution problem. |
| 05 | `05_bin_interpolation_fix` | Attempted fix for 04's sampling-resolution problem (interpolate dV/dQ across every bin a pulse spans); found it doesn't help and actively hurts intervals that already worked. |
| 06 | `06_cont_random_discharge_soc_sweep` | Continuous discharge + randomized C-rate (0.2-1.0C) + lower voltage cutoffs (LFP 1.5V/NMC 1.8V, a divergence from the cutoffs used elsewhere). Held up well unlike 04, since continuous discharge doesn't have the pulse protocol's sampling-resolution issue. |
| 07 | `07_real_lfp_nmc_test` | Evaluates the trained pipeline against real LFP/NMC lab data. |
| 08 | `08_pulse_protocol_archive` | Archived copy of the original GITT pulse-protocol scripts, kept because experiments 01 and 04 still import it directly. |
| 09 | `09_calce_data_integration` | Integrates real CALCE experimental discharge data (`data/calce_experiment_data/`) into the pipeline alongside synthetic simulations. |
| 10 | `10_constant_current_three` | Continuous discharge at three discrete, predefined C-rates (0.2C/0.5C/1.0C) per battery rather than one random draw. |
| 11 | `11_voltage_cutoff_comparison` | Isolates experiment 06's voltage-cutoff change from its C-rate randomization by comparing the original cutoffs (1.8V/2.3V) against 06's lower ones (1.5V/1.8V) on an otherwise-identical run. Surfaced a solver-termination artifact that inflates part of experiment 03's reported accuracy. |
| 12 | `12_exclude_termination_artifact_bins` | Quantifies how much of experiment 03's accuracy the artifact found in 11 actually accounts for, by re-training with the affected bins excluded. Only one of four SOC intervals turns out to be materially inflated. |
