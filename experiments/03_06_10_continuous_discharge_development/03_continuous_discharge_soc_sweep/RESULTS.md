# Continuous discharge + SOC sweep

## Motivation

Every experiment so far (including the SOC sweep in
`experiments/01_leakage_fix_20_percent_coverage/soc_sweep.py`) used the
GITT pulse protocol — a pulse of current followed by a 1-hour relaxation
rest, repeated. Stena's actual process won't look like that: it's a
continuous constant-current discharge with no relaxation at all. This
experiment replaces the pulse protocol with a continuous discharge and
re-runs the same style of SOC-availability sweep to see whether the
picture changes.

## What changed vs. the live pipeline

- **`simulate_batteries_continuous.py`**: the whole pulse/rest protocol
  (`TOTAL_PULSE_CAPACITY_AH`, `PULSE_DURATION`, `REST_DURATION`,
  `STEP_COUNTS`) is replaced by a single continuous discharge step,
  `Discharge at {current} A until {v_min} V`, at a fixed **0.6C** for
  every battery (scaled to each size tier's own target capacity), run all
  the way to the voltage cutoff (1.8V LFP / 2.3V NMC — same cutoffs as the
  live pipeline). Each battery now produces exactly one trace instead of
  five (one per pulse step count), so this runs far fewer solves overall.
  Everything else (chemistry pooling, capacity scaling, SOH, resistance/
  ambient-temperature modeling, `RANDOM_SEED`) is unchanged.
- **`feature_engineering_continuous.py`**: with no pulses there's no
  relaxed OCV point to anchor on, so dV/dQ is computed between **every
  consecutive pair of raw samples** instead of at pulse boundaries, and
  each 0.1V bin takes the **mean** of every point-to-point value that
  lands in it (a continuous trace puts many raw transitions in the same
  bin, unlike the pulse protocol's single relaxed value per bin). The
  bin-by-absolute-voltage scheme and the `>=20%` mutual-coverage leakage
  filter (see `experiments/01_leakage_fix_20_percent_coverage/RESULTS.md`)
  are otherwise identical.
- **`sweep_continuous_discharge.py`**: same 4 SOC intervals as the
  existing sweep, same fresh-subprocess-per-interval / incremental-results
  design. `ml_pipeline.py` is reused unchanged from the repo root.

## Results

| SOC Interval | RF Accuracy | XGB Accuracy | Surviving Bin(s) | Batteries | Runtime |
|---|---|---|---|---|---|
| 0.7-1.0 | 77.32% | 77.32% | `dV_dQ_V_3.3_3.2` | 485 | 1.1 min |
| 0.5-0.8 | 95.88% | 97.94% | `dV_dQ_V_3.3_3.2`, `dV_dQ_V_3.2_3.1` | 485 | 1.1 min |
| 0.3-0.6 | 97.94% | 98.97% | `dV_dQ_V_3.3_3.2`, `dV_dQ_V_3.2_3.1`, `dV_dQ_V_2.4_2.3`, `dV_dQ_V_2.3_2.2` | 485 | 1.1 min |
| 0.1-0.4 | 94.85% | 91.75% | `dV_dQ_V_3.2_3.1`, `dV_dQ_V_2.4_2.3`, `dV_dQ_V_2.3_2.2` | 484 | 1.1 min |

Full sweep: 4.5 minutes total, zero solve failures blocking any interval —
vs. ~40+ minutes for the equivalent pulse-based sweep, since each battery
now needs one continuous solve instead of five pulse-protocol solves.

## Interpretation

- **Same qualitative shape as the pulse-based sweep**: accuracy is worst
  at the highest-SOC interval (0.7-1.0) and much better once the sampled
  range moves lower — but the effect is sharper here (77% vs. the pulse
  sweep's 96.9% at the same interval), and the *number of usable bins
  changes with SOC range too*, not just their coverage percentages.
- **More shared signal appears at lower SOC, not less.** Two bins near
  2.3-2.4V (right at NMC's voltage floor) only pass the `>=20%`
  mutual-coverage filter for the two *lower* SOC intervals — meaning a
  continuous discharge starting from a high SOC essentially never reaches
  that region for one of the chemistries before running very long, while
  a discharge starting low spends comparatively more of its trace there,
  giving both chemistries a real, comparable presence. This is the
  opposite of what the pulse-based sweep showed (where all four intervals
  landed on the same single `3.4-3.3` bin).
- **No single fixed "the" bin survives across every SOC range** here
  either (`3.3-3.2` appears in 3 of 4 intervals but not the last; the
  2.3-2.4V pair appears in only 2 of 4) — reinforcing the same
  arbitrary-missingness concern raised for the Stena redesign: which
  voltage bins are even usable depends on where in the discharge curve a
  given capture happens to land, not on some universal, always-present
  feature.
- Practically: a continuous-discharge classifier trained and evaluated on
  a single SOC range that happens to include the 0.1-0.6 span looks very
  strong (95-99%); one restricted to the top of the curve (0.7-1.0) looks
  much weaker (77%) with only one usable bin. Any deployment plan needs to
  account for which part of the discharge curve Stena's captures will
  actually fall in — the model's real, available signal isn't uniform
  across the SOC range.
