# Phase 2: sim-to-real classification test

## Motivation

So far, all results in this project have been based on PyBaMM simulations. The goal, however, is to eventually use the classifier on real battery measurements. This means an important test is whether a model trained on simulated data can classify real batteries.

The original version of this experiment instead trained and tested the model using only real data — see `../01_real_vs_real_device_confound/RESULTS.md`, which found that design's 100% accuracy was a same-device/same-lab confound, not a chemistry-classification result. This phase redesigns the test properly: train on simulated data, test on real data (a genuine sim-to-real transfer test), which sidesteps the device-confound problem entirely by using real data on only one side of the split.

## Data

* **LFP:** `DownloadedData/ENSAYOSBATERIA/` contains data from one real LFP cell (LiFePO4 32700, 6000 mAh) with 579 discharge cycles. Each cycle consists of rest, a 6 A discharge to 2.5 V, and rest, with approximately 1 Hz sampling.
* **NMC:** `DownloadedData/Dataset_2_NCM_battery/` contains data from three real NMC cells tested at different temperatures (`CY25-05_1-#1`, `CY35-05_1-#1`, `CY45-05_1-#1`), with 1,782 cycles in total. The data comes from Zhu et al., *Nature Communications* (2022).
* **Synthetic data:** The four simulated datasets from experiment 03 were combined. They cover the SOC intervals 0.7–1.0, 0.5–0.8, 0.3–0.6, and 0.1–0.4. Combining them gives a wider voltage range, which is needed to cover the full discharge curves in the real data.

## Method

The synthetic and real data were first combined into one dataset and marked as either synthetic or real using a `DataKind` column. Features were then extracted once using `feature_engineering_continuous.py`.

This was done so that both datasets use exactly the same feature columns. If the features were extracted separately, the synthetic and real datasets could end up with different voltage bins.

After feature extraction, the data was separated again:

* **Training:** 996 synthetic LFP + 943 synthetic NMC batteries
* **Testing:** 577 real LFP + 1,782 real NMC cycles

The models were trained on **100% synthetic data** and tested on **100% real data**. No real data was used for training.

A total of 10 voltage bins survived the feature selection, covering `3.4–3.3 V` to `2.5–2.4 V`.

## Results

| Model         | Accuracy |
| ------------- | -------: |
| Random Forest |   24.84% |
| XGBoost       |   24.84% |

The real test set contains 577 LFP and 1,782 NMC cycles, corresponding to 24.5% LFP and 75.5% NMC.

Both models therefore performed almost exactly like a model that always predicts LFP. The classification report confirms this:

* **LFP recall:** 100%
* **NMC recall:** 1%

In other words, almost all real NMC batteries were classified as LFP.

## Interpretation

The results show that the classifier trained on PyBaMM simulations **does not transfer well to the real battery data** with the current pipeline.

The model performs well on simulated test data, but this performance does not carry over to real cells. The large difference suggests that the dV/dQ patterns learned from the simulations are different from those found in the real NMC data.

This result is also consistent with experiment 16, which tested whether a model trained on one real data source could generalize to another real data source. That experiment also showed very poor generalization.

Together, these experiments show that the current dV/dQ voltage bin features have not been shown to generalize beyond the type of data they were trained on.

This is important when interpreting the high accuracy reported in experiments 03, 06, 11, 12, 14, and 15. Those results were obtained using simulated data and therefore show how well the model works on data generated in the same simulation environment. They do **not** show that the model can classify real batteries.

## Improvement attempt #1: termination-artifact fix (no improvement)

Before trying fixes, real vs. synthetic mean dV/dQ was compared directly
per shared bin. LFP was well matched (both roughly -0 to -9 per bin), but
synthetic NMC's magnitudes were 10-100x larger than real NMC's:

| Bin | Real NMC | Synthetic NMC (baseline) |
|---|---|---|
| 3.0-2.9 | -3.36 | -107.75 |
| 2.8-2.7 | -3.30 | -351.85 |
| 2.7-2.6 | -5.24 | -53.33 |

This matched the shape of a bug already found and fixed elsewhere:
experiment 14's `exclude_final_transition` fix for a solver-termination
artifact (PyBaMM's event-triggered discharge cutoff produces one
oversized final step, exploding the dV/dQ estimate). That fix lives in
**root** `feature_engineering.py`, but this experiment's synthetic-side
extraction used `experiments/03_.../feature_engineering_continuous.py`, a
separate local copy that never received it — a strong, direct candidate.

`attempt_1_artifact_fix.py` reruns the identical pipeline with
only that one change (root `feature_engineering.py`,
`exclude_final_transition=True`, in place of
`feature_engineering_continuous.py`).

**Result: no accuracy change at all** (24.84%/24.84%, identical to the
baseline to the decimal). The fix did remove the most extreme outliers —
synthetic NMC's peak magnitude dropped from -351.85 to -53.33 — but the
*systematic* gap remains:

| Bin | Real NMC | Synthetic NMC (after fix) |
|---|---|---|
| 3.0-2.9 | -3.36 | -19.14 |
| 2.8-2.7 | -3.30 | -38.43 |
| 2.7-2.6 | -5.24 | -53.33 |

Synthetic NMC is still ~5-10x larger than real NMC across most bins, not
just in the artifact-driven extremes that got fixed. **The termination
artifact was a real but minor contributor to the mismatch, not the
dominant cause.** Something more fundamental makes PyBaMM's simulated NMC
discharge curve steeper (larger dV/dt) than these particular real NMC
cells' actual behavior across most of the voltage range — most likely a
genuine parameter mismatch (the PyBaMM NMC parameter sets used don't
represent these specific real cells well), not a feature-extraction bug.

## Improvement attempt #2: broader C-rate diversity (no improvement)

Attempt #1 ruled out the termination artifact as the dominant cause.
This attempt tests whether wider **C-rate** diversity narrows the
remaining gap: experiment 03's synthetic data (used so far) is a fixed
0.6C for every battery; experiment 06's already-simulated data
(`data/const_random_soc_*/`) draws C-rate uniformly from 0.2-1.0C per
battery, including rates below 0.6C that should produce smaller,
more-realistic dV/dQ magnitudes if C-rate variety is part of the gap.

`attempt_2_broader_diversity.py` combines experiment 03's +
experiment 06's raw data (8 SOC-interval datasets total, no
re-simulation) as the synthetic training set — roughly double the
training data (3,825 vs. 1,912 synthetic samples), same real test set,
same artifact fix applied.

**Result: no meaningful change.** RF 24.84% (identical), XGB 24.37%
(marginally *worse*, within noise). The magnitude comparison confirms
why — synthetic NMC barely moved despite doubling the training set with
wide C-rate variety:

| Bin | Real NMC | Synthetic NMC (artifact fix only) | Synthetic NMC (+ wider C-rate) |
|---|---|---|---|
| 3.0-2.9 | -3.36 | -19.14 | -24.41 |
| 2.8-2.7 | -3.30 | -38.43 | -44.74 |
| 2.7-2.6 | -5.24 | -53.33 | -50.82 |

**C-rate diversity is not the lever either.** Two candidate fixes have
now both failed to move the systematic ~5-10x magnitude gap: the
feature-extraction artifact (attempt #1) and C-rate variety (attempt #2).
By elimination, this points more strongly at the remaining hypothesis:
the PyBaMM NMC parameter sets themselves (Chen2020/Mohtat2020/OKane2022)
producing a systematically steeper simulated dV/dQ curve than these
specific real NMC cells, regardless of discharge rate — not a pipeline
bug, a genuine electrochemical parameter mismatch.

## Improvement attempt #3: per-NMC-parameter-set isolation (found it)

Attempts #1/#2 pointed by elimination at the NMC parameter sets
themselves. Root `simulate_batteries.py` draws each NMC battery's
parameter set uniformly from `Chen2020`/`Mohtat2020`/`OKane2022` (recorded
in the raw data's `Base_Parameter_Set` column) and pools all three
together — every test so far trained on that pooled mix.
`attempt_3_per_nmc_parameter_set.py` filters experiment 03's
already-simulated NMC rows to one parameter set at a time (keeping all
LFP rows, which only ever use one set, Prada2013) and reruns the
sim-to-real test three times.

**Result: the parameter sets are not equally mismatched — one of them
(Mohtat2020) is responsible for essentially the entire gap.**

| NMC parameter set | RF accuracy | XGB accuracy | Synthetic NMC dV/dQ range |
|---|---|---|---|
| Chen2020 | **83.81%** | 24.37% | -0.90 to -6.84 (close to real's -0.28 to -7.73) |
| Mohtat2020 | 24.84% | 24.84% | -4.08 to -53.33 (still enormous) |
| OKane2022 | **84.06%** | 24.37% | -1.08 to -18.41 (much closer to real) |

Random Forest trained on Chen2020 or OKane2022 *alone* jumps to ~84% —
up from the pooled baseline's 24.84%, a bigger single change than
anything else tried. Mohtat2020 alone reproduces the pooled failure
almost exactly (same 24.84%, same inflated magnitudes as the original
pooled run), meaning its badly-mismatched dV/dQ values were dominating
the pooled training distribution and dragging Chen2020/OKane2022's
otherwise-reasonable signal down with them.

**XGBoost did not improve with any parameter set** (stuck at
24.37-24.84% throughout) — investigated below.

## Improvement attempt #4: Chen2020+OKane2022 pooled (does not preserve the win)

The natural next step from attempt #3: train on Chen2020+OKane2022
pooled together, dropping only Mohtat2020, hoping to keep the ~84% result
with some parameter-set diversity restored.

**Result: it doesn't work — accuracy collapses back to 24.37%, both
models, reproduced independently two ways** (a custom sklearn pipeline
and root `ml_pipeline.py` on the same features, to rule out an
implementation bug before trusting the result). Confusion matrix: 1782 of
1782 real NMC cycles misclassified as LFP.

**Why: the ~84% single-parameter-set results were more fragile than they
looked.** Checking feature importances explains it. Chen2020 alone (84%):

| Bin | RF importance | NMC NaN rate |
|---|---|---|
| `3.4-3.3` | **36%** | 10.7% (well covered) |
| `3.1-3.0` | 17% | 94.6% (almost entirely imputed) |
| `3.2-3.1` | 16% | 63.5% (mostly imputed) |

Random Forest is relying overwhelmingly on `dV_dQ_V_3.4_3.3` — a bin
where synthetic NMC almost always has a real value (89% coverage) and
synthetic LFP almost never does (99.8% NaN, imputed to a near-constant).
This is a genuine physical difference (NMC's higher voltage range means
it reaches this bin; LFP's plateau rarely does) rather than an artifact,
but it's a **near-binary coverage signal**, not a rich dV/dQ-magnitude
comparison — most of the other bins are 60-100% NaN for synthetic NMC
even in the "working" individual-parameter-set runs, something the
earlier headline number didn't surface.

Pooling Chen2020+OKane2022 shifts the model's reliance *away* from that
reliable bin toward the noisier, heavily-imputed ones (`3.1-3.0`: 23%
importance, `3.2-3.1`: 22%, `3.4-3.3` down to 21%) — with more pooled
samples, the sparse bins' locally-large magnitude differences apparently
look more attractive to the tree-splitting criterion in aggregate, even
though they're built on far less reliable (mostly-imputed) data. The
result is a less robust boundary that fails on real data.

**XGBoost anomaly, explained**: XGBoost hits **100% training accuracy**
on Chen2020-alone's ~1,296-sample training set (severe overfitting to a
small, sparse feature set) yet predicts real NMC as LFP almost universally
(mean predicted P(NMC) for true real NMC: 0.057; 2,357 of 2,359 real
samples predicted LFP). Its top feature is the *same* `dV_dQ_V_3.4_3.3`
bin, even more concentrated than RF's (60% importance vs. RF's 36%) — but
XGBoost's boosted, sequential-residual-fitting splits appear to learn
decision thresholds tuned too specifically to the synthetic training
distribution's exact value range, which don't transfer. Random Forest's
bagged, averaged-across-many-trees structure is inherently more robust to
this kind of train/test distribution shift, even when using nearly the
same top feature — a plausible, general explanation, not confirmed by
further hyperparameter tuning here.

## Improvement attempt #5: magnitude-based outlier-jump exclusion (tried, harmful — discarded)

Attempt #3/#4 traced the remaining low-voltage-zone contamination to a
minority of NMC batteries (~13% of Mohtat2020's, vs. ~0-1% of
Chen2020's/OKane2022's) leaving *more than one* oversized solver step
near the voltage cliff — `exclude_final_transition` only drops the
single positionally-last transition, so these batteries still
contaminate the 2.4-2.9V bins.

Tested a generalization: instead of excluding by position, exclude any
transition whose `|dV|` is an outlier (`> K *` that battery's own median
`|dV|`) relative to that battery's own typical step size, regardless of
position or count. Tried on experiment 03's SOC 0.1-0.4 data (baseline
with `exclude_final_transition` alone: RF 73.68%/XGB 75.79% on the one
surviving bin, `dV_dQ_V_3.2_3.1`) with K = 3, 5, 10.

**Result: harmful, not just ineffective.** Every tested K destroyed that
bin entirely (zero surviving bins, both models `n/a`) — a total loss of
the one thing that worked, not a smaller improvement. Cause: LFP's
coverage in that bin was unaffected by K (83.5% throughout), but NMC's
collapsed with K (43.4% → exactly 20.0% at K=10, landing right at the
mutual-coverage threshold and getting the bin dropped). **The filter is
systematically biased against NMC specifically** — NMC's genuine dV/dQ
signal involves larger swings relative to its own flatter baseline than
LFP's more uniform steps, so a per-battery *relative*-magnitude
threshold can't distinguish "solver artifact" from "real chemistry
signal" and strips real NMC signal preferentially.

**Conclusion: `exclude_final_transition`'s position-only design is the
safer choice, not an incomplete stand-in for a better one** — it never
risks removing legitimate mid-curve signal. This attempt was fully
discarded (not merged; the branch was deleted) rather than kept
behind a flag, specifically so this finding doesn't get mistaken for a
validated option later. **Do not retry a relative-magnitude outlier
filter along these lines** — a real fix for the multi-jump near-cutoff
cases would need a genuinely different, proximity-to-cutoff-based
criterion instead, which remains untried.

## Not yet done

* **Revised assessment**: neither the per-parameter-set result (attempt
  #3) nor the pooled attempt (#4) is a robust fix. The
  individual-parameter-set ~84% numbers rest heavily on one sparse,
  near-binary coverage feature and don't survive combining two "good"
  sets together — treat that result as a fragile, not-yet-actionable
  finding rather than a validated improvement. Attempt #5's fix for the
  underlying multi-jump cause was actively harmful (see above) and was
  discarded, not merged.
* Understand *why* Mohtat2020 specifically diverges from these real cells
  electrochemically. Checked: essentially **no** synthetic NMC batteries
  (any parameter set) leave raw samples *inside* the 2.4-2.9V zone at all
  — the whole discharge curve jumps over it in one adaptive-solver step
  for 99-100% of Chen2020/OKane2022 batteries and 87% of Mohtat2020's.
  Mohtat2020 is the outlier only in that ~13% of its batteries leave a
  stray sample there, and those remaining points are still contaminated
  by the same near-cutoff oversized-step mechanism the `exclude_final_transition`
  fix (experiment 14) only partially addresses — evidently more than one
  oversized step can occur near the voltage cliff for a minority of
  batteries, and the fix only removes the single last one. Chen2020/
  OKane2022's apparent immunity is therefore likely coincidental (no data
  there to be wrong about) rather than genuinely better physics.
* A genuinely proximity-to-cutoff-based fix for the near-cutoff artifact
  (not the relative-magnitude approach attempt #5 showed is harmful) —
  not attempted.
* The real NMC data has much coarser and more variable sampling than the
  simulated data — not tested in this pass.
* Real LFP's C-rate (6A/6Ah = ~1C) remains mismatched vs. the synthetic
  0.6C/0.2-1.0C ranges tested, but LFP isn't the chemistry that's
  failing, so this wasn't prioritized.
* Wider SOH/resistance-factor range — experiment 06 uses the same range
  as experiment 03 (0.50-0.85), so attempt #2 tested C-rate diversity
  only, not SOH diversity. Would need new simulation, not attempted here.

