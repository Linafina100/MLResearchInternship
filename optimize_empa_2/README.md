# Optimized EMPA pipeline

Uses Polars and NumPy for vectorized features, partitioned Parquet instead of
intermediate CSVs, and parameter-aware caching. Simulations run in parallel with
one solver pass; evaluation trains each classifier once for the full SOC sweep.

Physical parameter defaults are unchanged. Sampling and noise sequences differ,
timestamps retain float64 precision, and feature selection uses training data only.

## Run

From this folder, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked
uv run --locked python simulate_batteries.py
uv run --locked python build_real_empa.py
uv run --locked python evaluate.py
uv run --locked python make_plot.py
```

The EMPA source defaults to `../data/Dataset-rocrate`; override with `EMPA_DIR`.
Defaults generate 498 synthetic traces. Raw data and feature caches go under
`../data/`; evaluation writes `sweep_results.json` and the plot goes in `plots/`.
Set `SOC_SWEEP = False` in `evaluate.py` for pooled results only, written to
`evaluation_results.json`; skip `make_plot.py` in that mode.

Cache keys include parameters, input contents, code, and core library versions.
After changing parameters, rerun the affected simulation/builder, then evaluation:
evaluation does **not** regenerate upstream data automatically. Set `USE_CACHE=0`
to force the invoked stage to recompute. Set `SIM_WORKERS=1` to reduce simulation
memory use; the default is up to four workers.
