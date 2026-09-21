# Phase 1: real LFP vs. real NMC classification -- and why it's a confound

## Motivation

Every result up to this point in the project came from PyBaMM
simulations. This was the first test on real, physically-measured
battery data: can the dV/dQ voltage-bin classifier tell LFP and NMC
cells apart when both sides are real cycling data (no synthetic data
involved at all)?

## Data

- **LFP**: `DownloadedData/ENSAYOSBATERIA/` -- a single real LFP cell
  (LiFePO4 32700, 6000 mAh), 579 discharge cycles.
- **NMC**: `DownloadedData/Dataset_2_NCM_battery/` -- 3 real NMC cells
  (`CY25-05_1-#1`, `CY35-05_1-#1`, `CY45-05_1-#1`, one per test
  temperature), 1782 cycles total (Zhu et al., *Nature Communications*
  2022).

## Test 1: LFP vs. NMC -- 100% accuracy

`evaluate_real_vs_real.py` combines both real datasets, extracts
continuous voltage-bin dV/dQ features
(`feature_engineering_continuous.py`, >=20% mutual-coverage filter), and
runs an 80/20 stratified group split (grouped by `Battery_ID`, i.e. by
cycle) through `ml_pipeline.py`.

| Model | Accuracy |
|---|---|
| Random Forest | 100.00% |
| XGBoost | 100.00% |

Reran and reproduced exactly on 2026-09-21: 2359 real cycles (LFP +
NMC), 10 surviving voltage bins, 472 held-out test samples, 100.00%/
100.00%.

At face value this looks like a clean chemistry-classification result.
It is not -- see the control below.

## Test 2: the control -- predict device ID instead of chemistry

`evaluate_device_id_control.py` keeps **only** the 3 real NMC cells (no
LFP, no chemistry difference at all), relabels each cycle's `Chemistry`
column with its **device identity** (`CY25-05_1-#1` / `CY35-05_1-#1` /
`CY45-05_1-#1`) instead of "NMC", and reruns the identical
feature-extraction + classification pipeline.

| Task | Accuracy |
|---|---|
| LFP vs. NMC (chemistry, Test 1) | 100.00% |
| NMC cell A vs. B vs. C (same chemistry, different devices, Test 2) | 100.00% |

Verified 2026-09-21: 1782 NMC cycles, 17 surviving voltage bins
(CY25: 193, CY35: 1124, CY45: 465 cycles), 356 held-out test samples,
100.00%/100.00% for both Random Forest and XGBoost, with per-class
precision/recall/f1 all 1.00.

Since there is no chemistry difference between the 3 cells in Test 2,
this accuracy cannot reflect a chemistry signal. It can only be picking
up **device/lab/protocol fingerprints**: sampling characteristics,
equipment noise floor, exact test temperature, and other measurement
artifacts specific to each cell's test rig. Since Test 1 and Test 2 use
the same feature representation and the same classifiers and get the
same accuracy, the most parsimonious explanation is that **Test 1 is
measuring the same kind of confound**, not genuine LFP-vs-NMC chemistry
separability.

## Why the confound is so severe here

- **Sample imbalance at the device level**: 1 physical LFP cell vs. 3
  physical NMC cells. Every LFP cycle comes from the same rig; the
  chemistry label is perfectly correlated with "which lab/equipment
  produced this cycle."
- **`StratifiedGroupKFold` only guards same-cycle leakage.** The
  train/test split groups by `Battery_ID` (so no single cycle's data
  leaks across the split), but it has no notion of "device" -- cycles
  from the same physical cell freely appear in both train and test, so
  a model can memorize that cell's fingerprint and still generalize
  "correctly" within this evaluation despite that being exactly the
  wrong kind of generalization.
- **Wildly different measurement setups**: the LFP data is uniformly
  sampled at 1 Hz; the NMC data is adaptively sampled (median dt ~16.4s).
  Any feature sensitive to sampling density or noise characteristics can
  trivially separate the two datasets without touching chemistry at all.

## Conclusion: three requirements for a valid future real-data test

1. **Multiple independent physical cells per chemistry**, not 1 vs. 3 --
   otherwise chemistry and device identity are confounded by
   construction.
2. **The same lab, equipment, and test protocol for both chemistries** --
   so sampling rate, noise floor, and protocol artifacts cannot serve as
   a shortcut signal.
3. **Train/test splits by physical cell, not by cycle** -- so a model
   cannot pass evaluation by memorizing a specific cell's fingerprint
   from cycles seen elsewhere in training.

See `../02_sim_to_real/RESULTS.md` for how this experiment was
subsequently redesigned around a genuine sim-to-real transfer test
(train on simulation, test on real data) instead, which sidesteps
requirement 1-2 above by using real data on only one side of the split.
