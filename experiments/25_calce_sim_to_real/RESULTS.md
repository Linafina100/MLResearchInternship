# Experiment 25: CALCE spot-check (n=4) — not a generalization claim

## Why this exists

The user asked whether the current best pipeline also works against a
third real dataset, CALCE (`data/calce_experiment_data/`), which earlier
experiments (07/09/16) touched briefly. Investigated before building
anything: CALCE is capped at **4 real cells total** — 2 LFP
(`A1-007`/`A1-008`, 1.1Ah nominal) and 2 NMC (`SP20-1`/`SP20-3`, 2.0Ah
nominal), each a single low-current OCV characterization discharge, not a
cycling/aging series. That ceiling can't be raised by better code — there
simply are no more CALCE files in the repo. This experiment answers a
narrow question only: does the CURRENT best pipeline (experiment 24's
SOH-range synthetic data, NMC diffusivity/10, LFP OCP rate=-3) still call
these specific 4 cells correctly, compared to what a prior local fix found
against the OLD synthetic pipeline? It is deliberately **not** presented as
third-dataset evidence the way [experiment 22](../22_empa_rocrate_sim_to_real/RESULTS.md)'s
199-cell EMPA test is — n=4 means one flipped prediction moves the score by
25 points.

## The parser bug (fixed here, not in experiment 09 itself)

`experiments/09_calce_data_integration/build_calce_dataset.py` (still
unfixed on this branch, verified via `git log`) has a column-mapping
collision: `'cap' in c_lower` matches both `Charge_Capacity(Ah)` and
`Discharge_Capacity(Ah)`, and the duplicate-column dedup that follows
silently keeps whichever column came first in the sheet — the
always-zero charge-capacity column for 3 of the 4 files, losing their
data. A local, unpushed fix for this already exists
(`git show 3328d42`, branch `verify-experiment-09-locally`, explicitly
marked "per instruction" as not to be merged — experiment 09 belongs to a
colleague). Rather than edit that experiment's files, this one carries the
same one-hunk fix forward into its own copy,
`build_calce_dataset.py` in this folder — match discharge-capacity terms
before falling back to a bare `'cap'` match, explicitly skipping
charge-capacity columns.

Verified after running: all 4 files now report real, sane capacity data
close to their nominal ratings (LFP: 1.06/1.09 Ah vs. 1.1 Ah nominal, SOH
0.967/0.989; NMC: 1.75/2.16 Ah vs. 2.0 Ah nominal, SOH 0.873/1.000) — no
Ah-scale rescaling needed here, unlike experiment 22's EMPA coin cells
(these CALCE cells are already Ah-scale, close to the synthetic side's own
1.2/2.0/3.5 Ah targets).

**One structural limitation, found by inspection, not assumed:** the
`SP20-3` NMC file's voltage never drops below 3.49V (range 3.49–4.19V) —
it never reaches any of the 14 bins the mutual-coverage filter keeps for
this run (all ≤3.3V), so every one of its features is entirely
median-imputed at prediction time. The other NMC file (`SP20-1`) covers
2.50–4.18V and has real coverage in most bins.

## Result: 2/4 correct — both LFP cells misclassified as NMC

Combined `data/24_soh_range_0.8_1.0_v1.5/` synthetic (498 batteries) with
the 4 real CALCE cells, ran root `feature_engineering.py`
(`min_chemistry_coverage=0.2`, 14 bins survive) and `ml_pipeline.py`
unchanged.

| File | Chemistry | RF prediction | RF confidence | XGB prediction | XGB confidence |
|---|---|---|---|---|---|
| A1-007-OCV-25-20120905.xlsx | LFP | NMC | 0.987 | NMC | 0.994 | 
| A1-008-OCV-25-20120905.xlsx | LFP | NMC | 0.987 | NMC | 0.994 |
| 11_16_2015 SP20-3 (no target-zone coverage) | NMC | NMC | 0.633 | NMC | 0.994 |
| 11_5_2015 SP20-1 | NMC | NMC | 1.000 | NMC | 0.995 |

Both models: **2/4 correct, raw accuracy 50.00%, balanced accuracy 50.00%,
LFP recall 0.000, NMC recall 1.000** — worse than the prior local fix's
75.00% (LFP 2/2, NMC 1/2) against the *old*, pre-diffusivity-fix
exp06-based synthetic data. The two pipelines got different cells wrong,
not the same ones: the old pipeline missed one NMC cell, the current one
misses both LFP cells (with very high, confidently-wrong probability).

## Why: real LFP's deep-bin dV/dQ magnitude is anomalously large here

Checking each real cell's feature values against the synthetic-side means
in the same bins (`ml_features.csv`, synthetic n=249 per chemistry)
explains the confident LFP misclassification. Synthetic LFP's deepest
bin (2.0–1.9V) averages **-16.3**; synthetic NMC's averages **-21.4** —
comparable order of magnitude, as designed. But the two real LFP cells'
own deep bins run **-109 to -256** in the 2.5–2.0V range — 7–16x larger
than anything synthetic LFP was trained on, and larger than synthetic
NMC's values too. On a scaled feature space, that pushes real LFP's
vector closer to the NMC region than the LFP region, which is exactly
what both models did. The real NMC cell with usable coverage (`SP20-1`,
-3.9 to -48) sits much closer to its own synthetic NMC range by
comparison.

One plausible explanation, not chased down further here: these CALCE
files are explicitly labeled "low current OCV test" — a near-quasi-static
characterization protocol, unlike either the synthetic training range
(0.1–0.2C) or any other real dataset in this project (the original set,
EMPA's ~0.07–1.3C). At very low current, per-timestep capacity
increments shrink while LFP's plateau-to-knee voltage drop stays a real
physical transition, which can inflate dV/dQ's magnitude well beyond what
a normal-rate discharge produces. This is a hypothesis consistent with
the numbers above, not a confirmed mechanism.

## Conclusion

This is a **spot-check that changed, not a validated third dataset.** The
current best pipeline gets 2 of these 4 specific real cells right,
against the OLD pipeline's 3 of 4 — neither number is statistically
meaningful on its own (n=4), and the two runs disagree on which cells are
hard, which is itself the clearest illustration of why: a single flipped
prediction is a 25-point swing either way. The most substantive finding
here isn't the accuracy number at all — it's that CALCE's low-current OCV
protocol appears to push real LFP's dV/dQ magnitude well outside the
range any of this project's synthetic training data (any experiment, not
just 24) has ever covered, which is a different and more specific
domain-shift problem than the C-rate/temperature caveats already
documented for EMPA in [experiment 22](../22_empa_rocrate_sim_to_real/RESULTS.md).
Not pursued further given the 4-sample ceiling — fixing this would mean
tuning to n=2 LFP cells, which is not a generalizable calibration target.
