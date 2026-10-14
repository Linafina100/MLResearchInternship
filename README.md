# MLResearchInternship

This project develops machine learning models to distinguish between two types of lithium ion batteries: LFP and NMC. It uses features extracted from battery discharge curves, based on how voltage changes as capacity decreases (\(\mathrm{d}V/\mathrm{d}Q\)).

The main goal is to test how well models trained on simulated battery data can classify real experimental data. Two models are compared: Random Forest and XGBoost.

The project originally used a GITT pulse discharge method. The main pipeline now simulates continuous constant current discharge, which better reflects how the intended real world testing equipment operates.

## Repository layout

```text
simulate_batteries.py      # Simulates battery discharge curves using PyBaMM
feature_engineering.py     # Extracts useful features from discharge curves
ml_pipeline.py             # Trains and evaluates Random Forest and XGBoost
data/                      # Raw data and generated outputs (not tracked by Git)
experiments/               # Numbered experiments testing different approaches
optimize_empa/             # Ongoing model optimization and pipeline development
```

## Main pipeline

The main pipeline creates simulated battery data, extracts features from the discharge curves, and uses these features to train machine learning models. The models can then be tested on real experimental data to evaluate how well they transfer from simulations to real batteries.

The pipeline can be run from start to finish using three scripts.

### 1. `simulate_batteries.py`

Simulates how LFP and NMC batteries discharge using PyBaMM. The simulations vary several battery conditions, including initial state of charge (SOC), state of health (SOH), ambient temperature, and internal resistance.

The default discharge rate is 0.6C. Each simulation runs until the battery reaches its chemistry specific cutoff voltage. The script saves the simulated discharge data and diagnostic plots.

### 2. `feature_engineering.py`

Processes the discharge curves to extract features describing how voltage changes as the battery discharges.

It calculates \(\mathrm{d}V/\mathrm{d}Q\), which describes how voltage changes relative to the amount of charge discharged. The values are grouped into voltage bins of 0.1 V, creating a consistent set of features for the machine learning models.

Bins that are only reached by one battery chemistry are removed to prevent the models from relying on voltage ranges that would reveal the battery type too easily. See Experiment 01 for more details.

### 3. `ml_pipeline.py`

Uses the extracted features to train and evaluate Random Forest and XGBoost classifiers.

It also aligns features when testing the models on separate experimental datasets, ensuring that the input data has the expected format. Trained models and related preprocessing information can optionally be saved for later use.

### Running the pipeline

Run the following commands in order, without additional arguments, to perform a default run:

```bash
# 1. Simulate battery discharge curves
python3 simulate_batteries.py

# 2. Extract features from the discharge curves
python3 feature_engineering.py

# 3. Train and evaluate the machine learning models
python3 ml_pipeline.py
```

## Dependencies

The project does not currently include a dependency lockfile. The pipeline uses the following Python libraries:

- `pybamm` for battery simulations
- `pandas` and `numpy` for data processing
- `scikit-learn` and `xgboost` for machine learning
- `matplotlib` and `seaborn` for plots
- `joblib` for saving trained models and related objects

## Data

Raw data, simulation results, extracted features, and plots are stored under `data/`. These files are excluded from Git and are generated or recreated as needed.

See `data/README.md` for an overview of the available data, what each folder contains, and which scripts generate the files.

## Experiments

The `experiments/` directory contains separate experiments that test different approaches to improving the pipeline. These include changes to the simulation settings, feature engineering, and model performance.

Experiments are organized into numbered folders, sometimes grouped by topic. Each experiment contains a `RESULTS.md` file explaining what was tested, what was changed, and what the results showed. Additional scripts contain the code used to run the experiment.

See `experiments/README.md` for an overview of the experiment groups and their purpose.

Experiment numbers may occasionally overlap because different people have created folders independently. Before starting a new experiment, check the existing folders with `ls experiments/` to avoid assigning a number that is already in use.
