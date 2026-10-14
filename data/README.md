# data/ layout

Gitignored, local-only, fully regenerable -- this file is the one
exception (`.gitignore` uses `data/*` + `!data/README.md`). Every
numbered folder is read back by a specific script at a hardcoded path;
renaming or moving one requires updating that script too (traced below,
not guessed from the name).

## 1. Raw downloaded datasets (not pipeline output)

| Folder | What it is |
|---|---|
| `Dataset-rocrate/` | The downloaded EMPA RO-Crate dataset -- 199 coin cells (`empa__ccidNNNNNN/`), each a `.bdf.csv`/`.bdf.parquet`/`.metadata.json` triple. |
| `calce_experiment_data/` | Raw downloaded CALCE OCV test spreadsheets (`.xls`/`.xlsx`). |

## 2. `experiments/` pipeline output

Organized by the same thematic groups as `experiments/` itself (see
`experiments/README.md`).

**`01_08_pulse_protocol_development/`**
| Folder | Produced by |
|---|---|
| `01_soc_0.1-0.4` (+3 SOC-interval siblings) | `01_leakage_fix_20_percent_coverage/soc_sweep.py` |
| `04_pulse_variable_soc_0.1-0.4` (+3) | `04_pulse_variable_discharge_soc_sweep/sweep_pulse_variable_discharge.py` |

**`03_06_10_continuous_discharge_development/`**
| Folder | Produced by |
|---|---|
| `03_continuous_soc_0.1-0.4` (+3) | `03_continuous_discharge_soc_sweep/sweep_continuous_discharge.py` |
| `06_const_random_soc_0.1-0.4` (+3) | `06_cont_random_discharge_soc_sweep/sweep_const_discharge_random.py` |

**`11_15_termination_artifact_investigation/`**
| Folder | Produced by |
|---|---|
| `11_cutoff_original`, `11_cutoff_lower` (+2 low-SOC variants) | `11_voltage_cutoff_comparison/compare_voltage_cutoffs.py` |
| `16_low_voltage_v1.5` | Producing experiment removed in an earlier consolidation (was `16_low_voltage_only_sim_to_real`) -- see git history if ever needed. Kept for local traceability only. |
| `17_high_soh_low_crate_v1.5` | `17_18_nmc_coverage_gap_investigation/17_high_soh_low_crate_sim_to_real/simulate_batteries_high_soh_low_crate.py` |

(The `16`/`17` rows above live physically next to `11`'s output on disk
but are produced by `17_18_nmc_coverage_gap_investigation/` -- grouped
here with the other termination/coverage diagnostics for readability,
not strictly by producing-group.)

**`20_21_diffusivity_ocp_tuning/`** (unmoved)
| Folder | Produced by |
|---|---|
| `20_soh_0.8_diffusivity_tuned_v1.5` | `20_nmc_diffusivity_tuning/simulate_batteries_diffusivity_tuned.py` |
| `21_soh_0.8_lfp_ocp_tuned_v1.5` | `21_lfp_diffusivity_tuning/simulate_batteries_lfp_ocp_tuned.py` (reused directly by experiments 22/23, no resimulation) |

**`22_29_sim_to_real_validation/`** (unmoved)
| Folder | Produced by |
|---|---|
| `22_real_empa_soh_0.8` | `22_empa_rocrate_sim_to_real/build_real_empa_dataset.py` |
| `23_real_empa_soc_sweep` | `23_empa_soc_sweep/build_real_empa_soc_sweep_dataset.py` |
| `24_soh_range_0.8_1.0_v1.5` | `24_soh_range_0.8_1.0_sim_to_real/simulate_batteries_soh_range.py` |
| `24_real_empa_soh_0.8_1.0` | `24_soh_range_0.8_1.0_sim_to_real/build_real_empa_soh_range_dataset.py` |
| `25_calce_real` | `25_calce_sim_to_real/build_calce_dataset.py` |
| `26_real_exp07_soc_sweep` | `26_soc_sweep_three_datasets/build_real_exp07_soc_sweep.py` |
| `26_real_empa_soc_sweep_soh_range` | `26_soc_sweep_three_datasets/build_real_empa_soc_sweep_soh_range.py` |
| `26_real_calce_soc_sweep` | `26_soc_sweep_three_datasets/build_real_calce_soc_sweep.py` |
| `26_soh_range_continuous_discharge_truncated_v1` | `26_soc_sweep_three_datasets/simulate_batteries_continuous_discharge_truncated.py` -- validation-only synthetic data (continuous-discharge-then-truncate); see that experiment's RESULTS.md for the rested-vs-continuous distinction |
| `27_real_snl_soc_sweep` | `27_snl_soc_sweep/build_real_snl_soc_sweep.py` -- SNL 18650 LFP/NMC cells |

**`30_31_snl_dataset_comparison/`**
| Folder | Produced by |
|---|---|
| `30_SNL_1` | `30_snl_vs_real/simulate_batteries_30.py` |
| `30_SNL_Real` | `30_snl_vs_real/Make_SNL_dataset_30.py` |

**`32_fixed_voltage_cutoff_soc_sweep/`** (unmoved, standalone)
| Folder | Produced by |
|---|---|
| `32_fixed_voltage_cutoff_synthetic_v1` | `simulate_batteries_continuous_discharge_truncated.py` (fixed-cutoff variant) |
| `32_real_empa_fixed_cutoff`, `32_real_exp07_fixed_cutoff`, `32_real_snl_fixed_cutoff` | this experiment's real-data build scripts, one per dataset |
| `32_lfp_negative_electrode_balance` | `data_generation/simulate_lfp_negative_electrode_balance.py` -- LFP-only resimulation with the validated negative-electrode capacity fix |
| `32_synthetic_lfp_balance_fix_combined` | `data_generation/combine_lfp_fix_with_baseline_nmc.py` -- combines the fix above with unmodified baseline NMC |

**`32_fixed_voltage_cutoff_soc_sweep/`** (this experiment's own namespace -- added when its per-SOC-point evaluators' raw output was moved out of `experiments/` to match every other experiment's convention)
| Folder | Produced by |
|---|---|
| `per_soc_raw/sim_and_real_raw_{exp07,EMPA,SNL}_soc_*.csv` + `per_soc_raw/features/` | `evaluation_baseline/evaluate_per_soc_point_models.py` |
| `per_soc_raw/sim_and_real_raw_ratiomed_{exp07,EMPA}_soc_*.csv` + `per_soc_raw/features_ratio_to_median/` | `evaluation_ratio_to_median/evaluate_per_soc_point_models_ratio_to_median.py` |
| `tmp_avg_dvdq/` | `diagnostics/plot_average_discharge_curves.py`'s scratch working dir |
| `tmp_scale_invariant_smoke/` | `diagnostics/smoke_test_scale_invariant_features.py`'s scratch working dir |

(A twelfth set, `sim_and_real_raw_balancefix_EMPA_soc_*.csv`, was deleted
during this move -- confirmed via repo-wide grep that no script anywhere
produces or reads files with that prefix; it was dead output from some
earlier, since-removed version of the evaluator.)

## 3. Orphaned: former `optimize_empa/` campaign data (producing folder deleted)

The original, completed `optimize_empa/` campaign (negative-electrode-
balance + ratio-to-median tuning) was deleted once its single validated
finding had been fully carried forward as the attempt-01 baseline of
the campaign now living at `optimize_empa/` (see section 4 -- that's a
different, later campaign that reused the freed folder name). Nothing
currently produces or reads these two folders.

| Folder | Produced by (deleted) |
|---|---|
| `optimize_empa_01_real_empa` | was `optimize_empa/experiment/01_ratio_to_median_baseline/data_generation/build_real_empa_soc_sweep_soh_range.py` |
| `optimize_empa_01_synthetic_baseline` | was that experiment's `data_generation/` scripts (negative-electrode-balance-fixed LFP + unmodified NMC) |

## 4. `optimize_empa/` campaign data (active, this branch)

| Folder | Produced by |
|---|---|
| `optimize_empa_low_soc_01_real` | attempt 01's `data_generation/build_real_empa.py` -- reused unchanged by every later attempt (real-data generation doesn't depend on the synthetic-side physics fixes) |
| `optimize_empa_low_soc_01_synthetic` | attempt 01's `data_generation/simulate_batteries.py` |
| `optimize_empa_low_soc_02_synthetic` | attempt 02's generator (`NEG_CAPACITY_FACTOR=0.7`) |
| `optimize_empa_low_soc_03_synthetic` | attempt 03's generator (`NEG_CAPACITY_FACTOR=0.45`, tuned) |

(Attempts 04/05 are evaluation-only changes -- they reuse attempt 03's
synthetic data and attempt 01's real data directly, no new folder. The
`optimize_empa_low_soc_*` data-folder names predate this campaign's code
folder being renamed from `optimize_empa_low_soc/` to `optimize_empa/` --
left unchanged per this project's data/ convention of not renaming
regenerable output just to track a code move.)

## 5. `optimize_empa/base_pipeline/` campaign data (colleagues' shared pipeline, documented here for completeness)

| Folder | What it is |
|---|---|
| `raw/`, `plots/`, `failures/` | `base_pipeline`'s default, unnamespaced output location (its scripts' `DATA_DIR` defaults straight to `data/`, not a per-campaign subfolder) -- `raw/` holds both its synthetic and real CSVs together. |

## 6. Cross-cutting / diagnostic (not experiment-specific)

| Folder | What it is |
|---|---|
| `default/` | Root pipeline scripts' (`simulate_batteries.py` etc.) own generic `RUN_LABEL` fallback when none is set. |
| `main_pipeline_<date>/` | Root pipeline's own dated diagnostic runs -- see `tools/make_main_pipeline_plots.py`. |
| `simulating_problems/` | Diagnostic plots, cross-cutting (not tied to one experiment). |

## Orphans

None currently tracked -- `archive/` (a pre-consolidation leftover whose
generating script was deleted) was removed as part of this document's
last cleanup pass. If a future folder stops being referenced by any
script, note it here rather than silently deleting it, so there's a
record of why it was kept or removed.
