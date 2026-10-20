# optimize_empa/

Active low-SOC optimization campaign built on `base_pipeline/`'s shared
EMPA SOC-sweep pipeline. Seven attempts so far (01-07); current best is
`experiment/06_pooled_model_full_soc_range/` -- one pooled model
trained on a genuinely full-SOC-range synthetic set, superseding
attempts 01-05's per-SOC specialization entirely (which also turned
out to be why Random Forest was collapsing to chance in every earlier
attempt). Attempt 07 confirms the result generalizes beyond tree
ensembles (MLP matches; an RBF-kernel SVM works but trails). See the
top-level `RESULTS.md` for the full attempt log.

(A separate, earlier, now-completed campaign used to live at this path
-- negative-electrode-balance + ratio-to-median tuning, whose single
validated finding was carried forward as this campaign's attempt 01
baseline. See `experiment/01_baseline_carryover/RESULTS.md`.)
