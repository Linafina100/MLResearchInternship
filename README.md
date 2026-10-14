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
ml_pipeline.py             # Trains + evaluates Random Forest / XGBoost on those features (writes models/ if run standalone -- not tracked, not kept up to date)
data/                      # Gitignored: all simulation/feature/plot output -- see data/README.md
experiments/               # Numbered, self-contained experiments -- see experiments/README.md
optimize_empa/             # Active low-SOC optimization campaign; base_pipeline/ is colleagues' shared EMPA pipeline
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

`experiments/` holds numbered, self-contained variations on the main
pipeline, grouped thematically into folders like
`20_21_diffusivity_ocp_tuning/` -- each leaf experiment has its own
`RESULTS.md` write-up, plus one or more scripts that either fork the root
pipeline's simulation/feature logic or reuse it directly and just sweep a
parameter. See `experiments/README.md` for the full index of groups and
what each one covers; numbering collides occasionally when multiple
people add a folder independently, so always check `ls experiments/` on a
fresh pull before assigning a new number.

## Data

All simulation/feature/plot output is gitignored and regenerated per run
under `data/`. See `data/README.md` for the full index of what each
folder contains and which script produces it.
