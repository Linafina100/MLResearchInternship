# Real LFP vs. NMC classification test

## Motivation

Every result so far in this project comes from PyBaMM simulations. This
experiment tests the approach on real, physically-measured battery data:
a real LFP cell's discharge cycles paired with real NMC cells' cycling
data, both already present in `DownloadedData/`.

## Data

- **LFP**: `DownloadedData/ENSAYOSBATERIA/` — a single real LFP cell
  (LiFePO4 32700, 6000mAh), 579 discharge cycle files (rest -> 6A
  discharge to 2.5V cutoff -> rest, ~1Hz sampling).
- **NMC**: `DownloadedData/Dataset_2_NCM_battery/` — 3 real NMC cells, one
  per test temperature (`CY25-05_1-#1`, `CY35-05_1-#1`, `CY45-05_1-#1`),
  1782 cycles total, from a published study (Zhu et al., *Nature
  Communications* 2022).

`parse_real_lfp.py` and `parse_real_nmc.py` reshape both into the
(`Time [s]`, `Voltage [V]`, `Capacity [A.h]`, `Chemistry`, `Variation_ID`)
format `experiments/03_continuous_discharge_soc_sweep/feature_engineering_continuous.py`
expects (unchanged, reused directly), with capacity derived by integrating
discharge current for LFP and taken directly from the NMC files' own
per-cycle `Q discharge/mA.h` column (confirmed to reset to 0 every cycle).
`ml_pipeline.py` (unchanged, repo root) then runs a plain train/test split
on the combined result — 2359 real batteries (577 LFP cycles + 1782 NMC
cycles survived the feature extraction).

## Result: 100% accuracy — and why that's not a real finding

| Model | Accuracy |
|---|---|
| Random Forest | 100.00% |
| XGBoost | 100.00% |

Given this session's whole methodology, a perfect score on the first try
is the number to be *most* suspicious of, not pleased with — so before
reporting this as "the pipeline works on real data," a direct control was
run: **relabel the 3 NMC cells by their own device identity (CY25 vs CY35
vs CY45) instead of by chemistry, and run the identical
feature-extraction + training pipeline.**

| Task | Accuracy |
|---|---|
| LFP vs. NMC (chemistry) | 100.00% |
| NMC cell A vs. B vs. C (same chemistry, different physical devices) | **100.00%** |

**The pipeline distinguishes the three same-chemistry NMC cells from each
other exactly as perfectly as it distinguishes LFP from NMC.** This
proves the "chemistry" result is not measuring chemistry at all — it's
picking up whatever makes each physical setup distinguishable, and
chemistry happens to be confounded with that in this dataset.

## Why the confound is so severe here

- **Sample size**: 1 physical LFP cell vs. 3 physical NMC cells. Every
  single LFP example comes from the exact same device, tested on the
  exact same rig, under the exact same fixed protocol (6A, 2.5V cutoff).
  There is no independent second LFP cell to check whether the "LFP
  signature" generalizes past this one unit.
- **The train/test split only guards against same-*cycle* leakage, not
  same-*device* leakage.** `Battery_ID` here is unique per cycle, so
  `StratifiedGroupKFold` correctly keeps any single cycle out of both
  train and test — but hundreds of *other* cycles from that same physical
  LFP cell (and the same physical NMC cells) are still in the training
  set whenever a given cycle is held out for test. The model can trivially
  memorize each specific device's fingerprint from its many other cycles.
- **Wildly different measurement setups.** LFP data is uniformly sampled
  at 1Hz (median dt = 1.0s); the NMC files have adaptive, much coarser
  sampling (median dt ~16.4s across a full cycle). Different labs,
  different cyclers, different protocols (a fixed 6A/2.5V rig vs. a
  temperature-controlled study at 25/35/45C) — any of these alone could
  give a classifier more than enough non-chemistry signal to separate the
  groups perfectly, and the NMC-vs-NMC control shows exactly that kind of
  signal is present and dominant.

## Conclusion

This experiment does **not** provide evidence that the voltage-bin
dV/dQ approach classifies real LFP vs. NMC cells correctly. It only shows
that with this specific data (1 device per one class, 3 devices from a
different lab for the other), *something* about the setups is trivially
separable — most likely equipment/protocol/sampling artifacts, not
chemistry. A real test of this kind needs, at minimum:

1. **Multiple independent physical cells per chemistry** (not 1 vs. 3),
   ideally several of each.
2. **The same lab, equipment, and test protocol for both chemistries** —
   varying only the chemistry, the way the synthetic simulations
   deliberately vary only chemistry while holding capacity, protocol, and
   noise model fixed.
3. **Splitting by physical cell, not by cycle** — every cycle from a
   given physical device should land entirely in train or entirely in
   test, never both, so the model cannot memorize per-device fingerprints
   across the split.

Until data meeting those conditions is available, this real-data
combination cannot be used to validate (or invalidate) the classification
approach.
