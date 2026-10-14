# MLResearchInternship

Classifies lithium-ion battery chemistry (LFP vs. NMC) from discharge curves via $\mathrm{d}V/\mathrm{d}Q$ voltage-bin features, benchmarking sim-to-real transfer from PyBaMM simulations to experimental cell data using Random Forest and XGBoost.

Originally built around a GITT pulse-discharge
protocol; the main pipeline now models a continuous constant-current
discharge instead, closer to how the intended real-world test equipment actually runs cells.

## Repository layout

```
simulate_batteries.py      # Simulates LFP/NMC discharge curves (PyBaMM)
feature_engineering.py     # Extracts dV/dQ voltage-bin features from simulated/raw discharge data
ml_pipeline.py             # Trains + evaluates Random Forest / XGBoost on those features 
data/                      # Gitignored: all raw data + simulation/feature/plot output -- see data/README.md
experiments/               # Numbered, self-contained experiments -- see experiments/README.md
optimize_empa/             # Active optimization + pipeline  --see optimize_empa/README.md
```

### Root pipeline
The root pipeline generates synthetic baseline data and can be executed out-of-the-box with no arguments for a single default run. The trained models can then be combined and evaluated against real-world experimental data for sim-to-real transfer validation.

`simulate_batteries.py`: Runs continuous discharge simulations (defaulting to 0.6C) across randomized LFP and NMC cell states (varying initial SOC, SOH, ambient temperature, and internal resistance) down to chemistry-specific cutoff voltages. It outputs raw time-series traces $\mathrm{d}V/\mathrm{d}Q$ alongside discharge-curve diagnostic plots.
`feature_engineering.py`: Computes continuous $\mathrm{d}V/\mathrm{d}Q$ derivatives, discretizes them into 0.1 V voltage bins, and filters out bins that only a single chemistry ever reaches to prevent trivial target leakage (see Experiment 01).
`ml_pipeline.py`: Consumes the engineered feature matrices to train and evaluate classifiers (Random Forest, XGBoost), handling feature alignment for holdout experimental sets and optionally serializing trained estimators, scalers, imputers, and metadata to models/.

Run directly with no arguments for a single default run:
```
# 1. Simulate continuous synthetic discharge profiles (PyBaMM)
python3 simulate_batteries.py
# 2. Extract and align 0.1V dV/dQ voltage-bin features
python3 feature_engineering.py
# 3. Train models and benchmark sim-to-real performance
python3 ml_pipeline.py
```

### Dependencies

No lockfile is currently checked in. The pipeline imports: `pybamm`,
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`,
`joblib`.

## Data

All simulation/feature/plot output is gitignored and regenerated per run
under `data/`. See `data/README.md` for the full index of what each
folder contains and which script produces it.


## Experiments

`experiments/` holds numbered, self-contained variations on the
pipeline, grouped thematically into folders like
`20_21_diffusivity_ocp_tuning/` -- each leaf experiment has its own
`RESULTS.md` write-up, plus one or more scripts that either fork the root
pipeline's simulation/feature logic or reuse it directly and just sweep a
parameter. See `experiments/README.md` for the full index of groups and
what each one covers; numbering collides occasionally when multiple
people add a folder independently, so always check `ls experiments/` on a
fresh pull before assigning a new number.

