The `data/` folder is intentionally excluded via `.gitignore` and does not appear on GitHub because the raw battery simulation outputs and empirical traces consist of dense, high-frequency time-series files. 

All raw simulation data, extracted feature tables, and diagnostic plots are generated deterministically by the provided scripts (`simulate_batteries.py` and `feature_engineering.py`), meaning they can be reproduced locally whenever needed while keeping the GitHub repository lightweight, modular, and focused purely on code.

### Directory Layout

```text
data/
├── Rocrate-dataset                              # The raw EMPA dataset
├── Arbitrary-dataset                            # The current version only uses EMPA's RoCrate
│
default/                                         # Runs where no specific environment (run_label) variable is defined
│       ├── raw/
│       │   └── synthetic_battery_data.csv
│       ├── features/
│       │   └── ml_features.csv
│       ├── failures/
│       │   └── simulation_failures.csv
│       └── plots/
│           └── continuous_discharge_plot.png       
│
└── <run_label>/                                 # Run-specific output directories (default: "default/")
    ├── real_empa_raw.csv                        # Processed empirical validation data (EMPA RO-Crate), output from `build_real_empa.py`, input in `evaluate.py`
    ├── raw
    │   └── advanced_synthetic_battery_data.csv  # Raw time-series discharge traces from single simulation runs, ouput from `simulate_batteries.py, input in `evaluate.py`
    ├── failures/
    │   └── simulation_failures.csv              # Solver errors and early-termination logs, ouput from `simulate_batteries.py`
    └── plots/
        └── continuous_discharge_plot.png       # Overview plots of generated V-t curves, output from `simulate_batteries.py`
```

 
| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATA_DIR` | `data` | Base directory for storing all simulation runs and artifacts. |
| `RUN_LABEL` | `default` | Experiment tag used to subfolder outputs (`data/<RUN_LABEL>/...`). |
| `OUTPUT_DATA_CSV` | `data/<RUN_LABEL>/raw/synthetic_battery_data.csv` | Explicit target path for raw simulation traces. |
| `FAILURE_LOG_CSV` | `data/<RUN_LABEL>/failures/simulation_failures.csv` | Filepath for logging PyBaMM solver error exceptions. |
| `DISCHARGE_PLOT_PNG` | `data/<RUN_LABEL>/plots/continuous_discharge_plot.png` | Destination for the discharge curve overview figure. |
