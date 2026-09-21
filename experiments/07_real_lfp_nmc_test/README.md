# Experiment 07: real LFP vs. NMC classification

## Original problem

Every other experiment in this project trains and evaluates the dV/dQ
voltage-bin classifier entirely on PyBaMM-simulated data. That leaves
the central question this project actually cares about untested: **does
any of it work on real, physically-measured battery data?**

Experiment 07 exists to answer that. It went through two designs, in
this order:

1. **Phase 1 — real vs. real** (`01_real_vs_real_device_confound/`):
   train and test on real LFP and real NMC data only, no simulation
   involved. Looked like a strong result (100% accuracy) until a control
   test showed it wasn't measuring chemistry at all.
2. **Phase 2 — sim-to-real** (`02_sim_to_real/`): the corrected design —
   train on simulated data, test on real data. This is the design that
   actually tests "does simulation-trained-classification transfer to
   real cells."

## Folder layout

```
07_real_lfp_nmc_test/
├── parse_real_lfp.py                        # shared: parses DownloadedData/ENSAYOSBATERIA/
├── parse_real_nmc.py                        # shared: parses DownloadedData/Dataset_2_NCM_battery/
├── 01_real_vs_real_device_confound/
│   ├── evaluate_real_vs_real.py             # real LFP vs. real NMC, plain train/test split
│   ├── evaluate_device_id_control.py        # control: predict device ID instead of chemistry
│   └── RESULTS.md
└── 02_sim_to_real/
    ├── evaluate_sim_to_real.py              # baseline: train on sim (exp03), test on real
    ├── attempt_1_artifact_fix.py            # improvement attempt #1
    ├── attempt_2_broader_diversity.py       # improvement attempt #2
    ├── attempt_3_per_nmc_parameter_set.py   # improvement attempt #3
    ├── attempt_4_chen_okane_pooled.py       # improvement attempt #4
    └── RESULTS.md                           # also documents attempt #5 (discarded, see below)
```

`parse_real_lfp.py` / `parse_real_nmc.py` stay at the top level because
both phases' scripts use them.

## Phase 1: the initial test, and why it doesn't mean what it looks like

`01_real_vs_real_device_confound/evaluate_real_vs_real.py` combines real
LFP (1 cell, 579 cycles) and real NMC (3 cells, 1782 cycles) data and
runs a plain 80/20 train/test split.

**Result: 100% accuracy**, both Random Forest and XGBoost (reproduced
2026-09-21).

**The crucial control test**: `evaluate_device_id_control.py` throws
away the LFP data and the chemistry label entirely, and instead asks the
classifier to tell the 3 real NMC cells apart *from each other* (device
identity, not chemistry — there is no chemistry difference among them).

**Result: also 100% accuracy** (reproduced 2026-09-21).

Since Test 2 has no chemistry signal to detect by construction, and gets
the same accuracy as Test 1, the conclusion is unavoidable: **the
classifier in Test 1 was mostly or entirely picking up per-device/
per-lab/per-equipment fingerprints — sampling rate, noise floor, test
protocol — not a genuine LFP-vs-NMC chemistry signal.** With 1 LFP cell
vs. 3 NMC cells from a completely different lab and measurement setup,
chemistry and "which rig produced this data" are almost perfectly
confounded, and the train/test split (grouped only by cycle, via
`StratifiedGroupKFold`) has no way to detect or prevent that. Full
details, numbers, and the confound mechanism: `01_real_vs_real_device_confound/RESULTS.md`.

## Phase 2: sim-to-real, the corrected design

Because Phase 1's real-vs-real design is fundamentally confounded by the
1-cell-vs-3-cell, different-lab setup, the experiment was redesigned:
train on **simulated** data (PyBaMM, from experiment 03), test on real
data. This sidesteps the device confound because real data now appears
on only one side of the split — there's no "predict which real device"
shortcut available when the training set has no real devices in it at
all.

**Baseline result: 24.84% accuracy** (both models) — collapsed to
predicting "LFP" almost always (LFP recall 1.00, NMC recall 0.01),
landing almost exactly on the naive always-predict-LFP floor. Four
follow-up improvement attempts were made:

| Attempt | Idea | Result |
|---|---|---|
| #1 | Fix a known solver-termination artifact in synthetic feature extraction | No accuracy change (removed extreme outliers, not the systematic gap) |
| #2 | Broader synthetic C-rate diversity | No meaningful change |
| #3 | Isolate each PyBaMM NMC parameter set individually | **Found it**: Chen2020/OKane2022 alone → RF ~84%; Mohtat2020 alone reproduces the failure |
| #4 | Pool just Chen2020+OKane2022 (the two "good" sets) | Collapses back to ~24% — the 84% result doesn't survive pooling; traced to a fragile, near-binary coverage feature, not real magnitude signal |
| #5 | Generalize the artifact fix to exclude any outlier-sized jump, not just the last one | **Harmful** — destroyed the one surviving working bin. Discarded entirely (branch deleted, not merged) |

Full method, numbers, and diagnosis for all five attempts:
`02_sim_to_real/RESULTS.md`.

**Bottom line for Phase 2**: sim-to-real transfer has not been
demonstrated to work on this data with this pipeline. The ~84%
single-parameter-set number is real but fragile (a sparse coverage
artifact, not a validated fix) and should not be cited as evidence the
approach generalizes.

## Suggestions for future real-data testing

Both phases point at the same three requirements a future real-data test
would need to actually isolate a chemistry signal instead of a confound:

1. **Multiple independent physical cells per chemistry** — not 1 vs. 3.
   Every chemistry needs enough independent devices that "chemistry" and
   "which specific cell" aren't perfectly correlated.
2. **The same lab, equipment, and test protocol for both chemistries** —
   so sampling rate, noise floor, and protocol artifacts can't serve as
   a shortcut signal that substitutes for genuine chemistry
   differences.
3. **Train/test splits by physical cell, not by cycle** — grouping only
   by cycle (as `StratifiedGroupKFold` on `Battery_ID` does here) still
   lets a model see many cycles from the same physical device across
   both train and test, so it can memorize that device's fingerprint
   and pass evaluation without learning anything that generalizes to a
   *different* physical cell of the same chemistry.

Phase 2's sim-to-real redesign satisfies requirement 1-2 by construction
(no real device appears more than once, and there's no "other lab" to
confound with) but still hasn't produced a working classifier — so the
remaining, more fundamental gap is a genuine synthetic-vs-real dV/dQ
magnitude mismatch (see `02_sim_to_real/RESULTS.md`'s diagnosis),
independent of the confound issue Phase 1 uncovered.
