# Real LFP vs. NMC classification test: sim to real

## Motivation

So far, all results in this project have been based on PyBaMM simulations. The goal, however, is to eventually use the classifier on real battery measurements. This means an important test is whether a model trained on simulated data can classify real batteries.

The original version of this experiment instead trained and tested the model using only real data. This was not a sim to real test, so the experiment was redesigned to correctly evaluate how well simulated data transfers to real data.

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

`evaluate_sim_to_real_artifact_fix.py` reruns the identical pipeline with
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

`evaluate_sim_to_real_broader_diversity.py` combines experiment 03's +
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
`evaluate_sim_to_real_per_nmc_parameter_set.py` filters experiment 03's
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
24.37-24.84% throughout) — worth a follow-up in its own right, not
explained here; Random Forest is the one that benefits.

## Not yet done

* **Confirmed, actionable**: drop or down-weight Mohtat2020 from the
  training mix (or train on Chen2020+OKane2022 pooled together, avoiding
  Mohtat2020 specifically) and see if that combination preserves the
  ~84% RF result while keeping some parameter-set diversity — not yet
  tried, the natural next step.
* Why XGBoost doesn't benefit the way RF does from the same
  Chen2020/OKane2022-only data — unexplained, worth investigating
  separately (default hyperparameters interacting poorly with a smaller,
  ~1300-sample training set is one guess, not verified).
* Understand *why* Mohtat2020 specifically diverges from these real
  cells electrochemically (vs. Chen2020/OKane2022, which don't) — not
  investigated at the parameter level here, only empirically detected.
* The real NMC data has much coarser and more variable sampling than the
  simulated data — not tested in this pass.
* Real LFP's C-rate (6A/6Ah = ~1C) remains mismatched vs. the synthetic
  0.6C/0.2-1.0C ranges tested, but LFP isn't the chemistry that's
  failing, so this wasn't prioritized.
* Wider SOH/resistance-factor range — experiment 06 uses the same range
  as experiment 03 (0.50-0.85), so attempt #2 tested C-rate diversity
  only, not SOH diversity. Would need new simulation, not attempted here.

