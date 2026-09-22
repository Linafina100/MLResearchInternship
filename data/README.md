# data/ layout

Gitignored, local-only, fully regenerable. Numbered subfolders are
prefixed with the `experiments/NN_.../` number that produces them, so
it's clear at a glance which experiment made which data. Traced from
each folder's actual generating code, not guessed from the name.

| Folder | Produced by |
|---|---|
| `01_soc_0.1-0.4` (+3 SOC-interval siblings) | `experiments/01_leakage_fix_20_percent_coverage/soc_sweep.py` |
| `03_continuous_soc_0.1-0.4` (+3) | `experiments/03_continuous_discharge_soc_sweep/sweep_continuous_discharge.py` |
| `04_pulse_variable_soc_0.1-0.4` (+3) | `experiments/04_pulse_variable_discharge_soc_sweep/sweep_pulse_variable_discharge.py` |
| `06_const_random_soc_0.1-0.4` (+3) | `experiments/06_cont_random_discharge_soc_sweep/sweep_const_discharge_random.py` |
| `11_cutoff_original`, `11_cutoff_lower` (+2 low-SOC variants) | `experiments/11_voltage_cutoff_comparison/compare_voltage_cutoffs.py` |
| `16_low_voltage_v1.5` | one-off run of root `simulate_batteries.py` with `LFP_LOWER_CUTOFF`/`NMC_LOWER_CUTOFF`/`RUN_LABEL` overridden for `experiments/16_low_voltage_only_sim_to_real/` |
| `17_high_soh_low_crate_v1.5` | `experiments/17_high_soh_low_crate_sim_to_real/simulate_batteries_high_soh_low_crate.py` |
| `20_soh_0.8_diffusivity_tuned_v1.5` | `experiments/20_nmc_diffusivity_tuning/simulate_batteries_diffusivity_tuned.py` |
| `21_soh_0.8_lfp_ocp_tuned_v1.5` | `experiments/21_lfp_diffusivity_tuning/simulate_batteries_lfp_ocp_tuned.py` (still reused directly by experiments 22/23, no resimulation) |
| `22_real_empa_soh_0.8` | `experiments/22_empa_rocrate_sim_to_real/build_real_empa_dataset.py` |
| `23_real_empa_soc_sweep` | `experiments/23_empa_soc_sweep/build_real_empa_soc_sweep_dataset.py` |
| `24_soh_range_0.8_1.0_v1.5` | `experiments/24_soh_range_0.8_1.0_sim_to_real/simulate_batteries_soh_range.py` |
| `24_real_empa_soh_0.8_1.0` | `experiments/24_soh_range_0.8_1.0_sim_to_real/build_real_empa_soh_range_dataset.py` |
| `archive/soh_0.8_fixed_v1.5` | orphaned -- likely a pre-consolidation exp20 Part A/B intermediate; its generating script was deleted when experiments 20/21 were consolidated. Nothing references it; safe to delete entirely if disk space is needed. |

**Not experiment-specific, left unnumbered:**
- `default` -- root scripts' (`simulate_batteries.py` etc.) own generic `RUN_LABEL` fallback, not tied to one experiment.
- `main_pipeline_<date>` -- root pipeline's own dated diagnostic runs (see `make_main_pipeline_plots.py`).
- `Dataset-rocrate`, `calce_experiment_data` -- real (not synthetic) datasets.
- `simulating_problems` -- diagnostic plots, cross-cutting.

Every numbered folder above is read back by a specific experiment's
evaluate/build script at a hardcoded path -- renaming or moving one
requires updating that script too (see each producer file's
`RUN_LABEL`/`SYNTHETIC_RAW_CSV`/`REAL_RAW_CSV` constants).
