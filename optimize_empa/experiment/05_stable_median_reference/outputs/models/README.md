# outputs/models/

Per-`Initial_SOC`-bucket models from this experiment's own `run_evaluation.py`
(whichever of RF/XGBoost scored higher balanced accuracy at that bucket --
see each bucket's `metadata.json`). Not one global model: this pipeline
trains and evaluates a fresh pair per SOC bucket, since accuracy varies
sharply by how deep the real discharge window goes (see the top-level
`RESULTS.md`).

Each `soc_<X>/` folder is self-contained:

| File | What it is |
|---|---|
| `classifier.joblib` | The winning model for this bucket |
| `scaler.joblib`, `imputer.joblib`, `label_encoder.joblib` | This bucket's fitted preprocessing (clip -> impute -> scale, same order as training) |
| `clip_bounds.json` | The 1st/99th-percentile clip bounds learned from this bucket's own training data |
| `feature_names.json` | `bin_cols` (the dV/dQ voltage bins this bucket uses) and `median_reference_bins` (the fixed bins the ratio-to-median transform divides by -- apply this transform to new data BEFORE scaling) |
| `metadata.json` | Which model won, its balanced accuracy / per-chemistry recall, and whether this bucket was SOC-matched or extrapolated |

To reuse a bucket on new data: replicate `apply_ratio_to_median` from
`run_evaluation.py` using that bucket's `median_reference_bins`, clip
with `clip_bounds.json`, then impute -> scale -> predict with the
saved artifacts, in that order.

Regenerate with `python3 run_evaluation.py` (writes here by default;
set `SAVE_MODELS=0` to skip).
