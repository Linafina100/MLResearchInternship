# experiments/

Numbered, self-contained experiments on the main pipeline, grouped
thematically. Each leaf folder has its own `RESULTS.md`. See the root
`README.md` for the project overview.

| Folder | Covers | Theme | Wiki |
|---|---|---|---|
| `01_08_pulse_protocol_development/` | 01,02,04,05,08 | Original GITT pulse-discharge protocol: leakage fix, NaN fix, variable-rate problem, archived pulse code | Wiki:  |
| `03_06_10_continuous_discharge_development/` | 03,06,10 | Continuous constant-current discharge protocol that superseded pulse (became the main pipeline) | Wiki: |
| `07_09_19_early_real_data_validation/` | 07,09,19 | Early real-lab-data evaluation (LFP/NMC test rig, CALCE, EMPA) preceding the systematic sim-to-real series | Wiki: |
| `11_15_termination_artifact_investigation/` | 11,12,13,14,15 | Solver-termination artifact: discovery, quantification, fixes, and post-fix hyperparameter tuning | Wiki: |
| `17_18_nmc_coverage_gap_investigation/` | 17,18,18 | Why synthetic NMC data barely reaches the real target voltage zone | Wiki: |
| `20_21_diffusivity_ocp_tuning/` | 20,21 | NMC diffusivity fix + LFP OCP tail calibration -- first genuine sim-to-real signal | Wiki: |
| `22_29_sim_to_real_validation/` | 22-29 | Systematic validation across EMPA/SNL/CALCE real datasets | Wiki: |
| `30_31_snl_dataset_comparison/` | 30,31 | SNL vs. real-data comparison and curve visualization | Wiki: |
| `32_fixed_voltage_cutoff_soc_sweep/` | 32 | Per-SOC specialized models + negative-electrode-balance fix + ratio-to-median features -- fed into `optimize_empa/` | Wiki: |

Numbering has one intentional gap (16: the producing experiment was
removed during an earlier consolidation, see `data/README.md` and the duplicate "18" prefix is resolved by nesting both
under the same parent group rather than renumbering either one. 

Additional information about the decisions of the project can be found in the wiki.
