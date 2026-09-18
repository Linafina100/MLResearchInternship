# Real LFP vs. NMC classification test: sim-to-real

## Motivation

Every result in this project up to this point comes from PyBaMM
simulations. The point of building a classifier from simulated discharge
curves is to eventually apply it to real, physically-measured cells — so
the test that actually matters is: **train on simulated data, test on
real data.** This experiment's original version instead combined the real
LFP and NMC data and ran a plain train/test split *within* that real
data — not a sim-to-real transfer test, by its own prior docstring. That
wasn't the intended design. This is the corrected version.

## Data

- **LFP**: `DownloadedData/ENSAYOSBATERIA/` — a single real LFP cell
  (LiFePO4 32700, 6000mAh), 579 discharge cycle files (rest -> 6A
  discharge to 2.5V cutoff -> rest, ~1Hz sampling).
- **NMC**: `DownloadedData/Dataset_2_NCM_battery/` — 3 real NMC cells, one
  per test temperature (`CY25-05_1-#1`, `CY35-05_1-#1`, `CY45-05_1-#1`),
  1782 cycles total, from a published study (Zhu et al., *Nature
  Communications* 2022).
- **Synthetic training data**: experiment 03's four already-simulated
  SOC-interval raw datasets (`data/continuous_soc_{0.7-1.0,0.5-0.8,
  0.3-0.6,0.1-0.4}/raw/`), concatenated for full discharge-range coverage
  — a single narrow SOC interval wouldn't span the same voltage range the
  real cells' full discharge cycles do. No re-simulation needed.

## Method

`evaluate_real_data.py` builds one combined raw CSV (synthetic + real,
tagged with a `DataKind` column) and extracts features **once** via
`feature_engineering_continuous.py` (unchanged, `>=20%` mutual-coverage
filter), so the synthetic training set and real test set share the exact
same surviving feature columns — extracting them separately could pick
different bins and break the sim-to-real evaluation. The resulting
feature table is then split back into a synthetic CSV and a real CSV, and
`ml_pipeline.run_ml_pipeline(synthetic_csv=..., real_csv=...)` trains on
100% of the synthetic set and tests on 100% of the real set — no
synthetic data used for testing, no real data used for training.

10 bins survived (`3.4-3.3V` down to `2.5-2.4V`): 996 synthetic LFP + 943
synthetic NMC batteries for training, 577 real LFP + 1782 real NMC cycles
for testing.

## Result: near-total failure to generalize

| Model | Accuracy |
|---|---|
| Random Forest | 24.84% |
| XGBoost | 24.84% |

For context: real test data is 577 LFP / 1782 NMC (24.5% / 75.5%), so
"always predict NMC" would score 75.5%, and "always predict LFP" would
score 24.5%. **Both models landed almost exactly on the "always predict
LFP" floor** — the classification report confirms it directly: LFP recall
is 1.00 (100% of real LFP correctly identified) but NMC recall is 0.01
(99% of real NMC misclassified as LFP). Neither model is doing anything
resembling chemistry classification on real data; both have collapsed to
predicting the same class almost regardless of input.

## Interpretation

**A classifier trained purely on PyBaMM-simulated dV/dQ features does not
transfer to real, physically-measured cells for this pipeline as
currently built.** This isn't a subtle effect — it's a near-complete
collapse to a degenerate single-class predictor, worse than even a naive
majority-class baseline on the harder class. The simulated LFP and real
LFP evidently produce similar enough dV/dQ signatures in these bins that
the model defaults to "LFP" whenever it sees anything resembling either,
but real NMC's actual dV/dQ signature in this feature space looks nothing
like simulated NMC's.

This is consistent with, and now directly confirms in the sim-to-real
direction, what experiment 16 already found in the real-to-real direction
(training on one real source and testing on another real source also
collapsed to at-or-below-chance accuracy — see
`experiments/16_combine_real_lfp_nmc_calce/RESULTS.md`). Between the two
results, there is now no direction — sim-to-real, or real-source-to-real-
source — in which this project's dV/dQ voltage-bin classifier has been
shown to generalize outside the exact distribution it was trained on.

**This is the most consequential negative result in the project so far.**
Every synthetic-only accuracy number reported elsewhere (experiments
03/06/11/12/14/15, all 74-100% depending on interval and fix) describes
how well the model does on more PyBaMM simulations drawn from the same
generative process it was trained on — not on anything resembling a real
cell. Whatever is driving those numbers, it isn't (yet) shown to be
something that would work on Stena's actual measurements.

## Not yet done

- Diagnose *why* synthetic and real dV/dQ signatures diverge this much
  for NMC specifically — likely candidates: PyBaMM's SPM parameter sets
  don't match these particular real cells' true electrochemical
  parameters, the real NMC data's much coarser/adaptive sampling
  (~16.4s median vs. LFP's 1Hz) interacts badly with the fixed 0.1V
  binning, or the real cells' C-rates/protocols differ substantially from
  the simulated 0.6C. Not diagnosed here.
- Try training on a wider variety of simulated conditions (more NMC
  parameter sets, wider C-rate range, wider SOH range) to see whether
  broader synthetic diversity narrows the sim-to-real gap at all, or
  whether the gap is structural.
- This result, combined with experiment 16's, suggests the project's
  current real-data evidence doesn't yet support claiming the
  dV/dQ-voltage-bin approach classifies real chemistry at all — worth
  surfacing prominently before any further work assumes the synthetic
  accuracy numbers translate to real deployment.
