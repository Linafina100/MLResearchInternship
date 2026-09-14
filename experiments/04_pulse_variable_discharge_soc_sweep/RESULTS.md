# Pulse protocol with randomized discharge rate + SOC sweep

## Motivation

Every pulse-based experiment so far derives the per-pulse current from a
fixed total discharge capacity (`TOTAL_PULSE_CAPACITY_AH=0.6Ah` spread
evenly across `n_steps` pulses) -- current is a function of step count
only, never a property of the battery itself. This experiment keeps the
GITT pulse protocol (same timing: 30min pulse + 1h rest per step) but
makes the discharge rate a genuine per-battery random draw instead, run
alongside the existing 4-interval SOC-availability sweep.

## What changed vs. the live pipeline

- **`simulate_batteries_variable_pulse.py`**: each battery draws its own
  `pulse_c_rate = random.uniform(0.1, 0.5)` once per variation (same
  style as the existing SOC/SOH per-battery randomization), and
  `current_a = pulse_c_rate * target_ah` is used identically across all 5
  step counts for that battery -- unlike the live script, where current is
  re-derived per step count to hold total discharged capacity constant at
  0.6Ah regardless of `n_steps`. The random rate is recorded in the raw
  CSV as `Pulse_C_Rate` for transparency but confirmed to never reach the
  engineered feature CSV (not one of `feature_engineering.py`'s carried-
  through optional columns).
- **`feature_engineering.py` and `ml_pipeline.py` are reused unchanged**
  from the repo root -- pulse timing (which is all the step-boundary
  detection depends on) is untouched, and `ml_pipeline.py` is
  feature-set-agnostic.
- Same 4 SOC intervals, same sweep structure as
  `experiments/01_leakage_fix_20_percent_coverage/soc_sweep.py`.

## Results

| SOC Interval | RF Accuracy | XGB Accuracy | Surviving Bin(s) | Batteries | Runtime |
|---|---|---|---|---|---|
| 0.7-1.0 | 81.61% | 81.61% | `dV_dQ_V_3.4_3.3` | 433 | 15.8 min |
| 0.5-0.8 | — | — | none | 409 | 81.1 min |
| 0.3-0.6 | — | — | none | 358 | 10.9 min |
| 0.1-0.4 | — | — | none | 202 | 11.1 min |

Full sweep: 118.8 minutes total (vs. ~40 min for the original fixed-rate
pulse sweep) -- interval 2 alone took 81 minutes, likely from PyBaMM
solver difficulty on some higher-current/degraded-cell combinations that
the fixed-rate protocol never produced. Battery counts also decline with
SOC (433 -> 202) as lower-SOC + higher-current draws hit the voltage
cutoff, or fail to solve, more often.

## Interpretation

- **Randomizing discharge rate collapses the sweep's signal almost
  entirely.** Only the highest-SOC interval retains any voltage bin that
  passes the `>=20%` mutual-coverage filter at all; every other interval
  comes back with **zero usable bins** (`ml_pipeline.py`'s zero-feature
  guard reports this plainly rather than crashing -- see
  `experiments/01_.../RESULTS.md` for that fix's origin). This is a much
  more severe collapse than either of the other two sweeps this session:
  the original fixed-rate pulse sweep had *some* signal at every interval,
  and the continuous-discharge sweep too.
- **Why this makes sense**: dV/dQ's voltage bins are keyed on *absolute
  terminal voltage*, which includes the IR-drop contribution from
  whatever current is flowing. Different batteries now discharge at
  substantially different currents (0.1C-0.5C, a 5x spread), so the same
  physical state-of-charge point lands at different measured voltages
  for different batteries purely because of their randomly-drawn rate --
  smearing what used to be a consistent per-chemistry voltage signature
  across a wider, noisier voltage range. At high SOC there's apparently
  still enough of a physically distinct region (`3.4-3.3V`) that survives
  this smearing; at lower SOC, where the usable voltage range was already
  narrower to begin with (per the original SOC sweep's findings), the
  added rate-driven voltage spread is enough to wipe out any bin both
  chemistries reliably land in.
- **Practical takeaway**: a classifier relying on absolute-voltage dV/dQ
  bins is not just sensitive to *which* SOC range a real cell happens to
  be captured in (already shown by the other two sweeps) but also to
  *how consistent the test's discharge rate is* -- if Stena's actual
  testing equipment doesn't apply a tightly controlled, repeatable
  current, this feature scheme may have very little to work with outside
  a narrow SOC band. This reinforces the case (from the continuous-
  discharge redesign discussion) for features less sensitive to IR-drop
  variation than a fixed absolute-voltage bin grid.
