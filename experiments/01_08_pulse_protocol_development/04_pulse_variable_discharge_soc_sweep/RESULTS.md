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
- **`ml_pipeline.py` is reused unchanged from the repo root** (it's
  feature-set-agnostic); the pulse-based `feature_engineering.py` is
  shadowed in from `experiments/07_pulse_protocol_archive/`, since root's
  own `feature_engineering.py` is now the continuous-discharge version.
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

## Root cause: why does varying discharge rate collapse the signal?

The "smearing" explanation above was a hypothesis, not yet confirmed. A
follow-up diagnostic against the raw/feature data (no re-simulation
needed) found the actual mechanism, and it's more specific than smearing:

**1. Which bin flips, and where.** In the 0.7-1.0 interval (worked),
`dV_dQ_V_3.4_3.3` has 63.5% LFP coverage / 28.3% NMC coverage -- both
comfortably clear the 20% filter. In the 0.5-0.8 interval (failed), the
*same* bin's LFP coverage collapses to 9.5% (NMC barely moves, 31.3%) --
LFP alone drops below threshold and the bin gets dropped, leaving zero
surviving bins for the whole interval.

**2. What separates LFP batteries that still hit that bin from those that
don't**, within the 0.5-0.8 interval:

| | mean Pulse_C_Rate | max Pulse_C_Rate | mean Initial_SOC |
|---|---|---|---|
| Hit `3.4-3.3V` (23 batteries) | 0.167 | 0.227 | 0.761 |
| Missed it (220 batteries) | 0.310 | 0.494 | 0.638 |

Starting SOC matters (consistent with the SOC-only sweep's already-known
effect), but the C-rate split is sharper and compounds with it: batteries
that hit the bin never drew a C-rate above 0.227, while the missed group
averages 0.31 and reaches nearly 0.5.

**3. Direct mechanism, confirmed**: mean absolute voltage change between
consecutive relaxed pulse points (LFP, same interval):

| C-rate bucket | mean \|dV\| per step | n |
|---|---|---|
| 0.1-0.2C | 0.094 V | 57 |
| 0.2-0.3C | 0.119 V | 56 |
| 0.3-0.4C | 0.119 V | 50 |
| 0.4-0.5C | 0.231 V | 30 |

Per-pulse voltage steps more than double from the lowest to the highest
C-rate bucket, and at the top end they're comparable to or larger than
the 0.1V bin width itself.

**Conclusion**: this is a sampling-resolution problem, not smearing. The
pulse protocol only samples voltage once per pulse (the relaxed point).
At higher current, each pulse discharges more capacity, so consecutive
relaxed samples land farther apart in voltage -- a fixed 0.1V-wide bin
increasingly falls *between* two samples rather than catching one,
regardless of whether the chemistry "visits" that voltage region at all.
This directly lowers how often any given battery registers a value in a
specific bin, scaling with C-rate, and it compounds with the already-known
SOC-range effect (fewer batteries even reaching the relevant voltage zone)
to push bins below the 20% mutual-coverage threshold at far less extreme
SOC restriction than the fixed-rate protocol needed (0.5-0.8 fails here
vs. 0.3-0.6 for the fixed-rate sweep).

This also points to the fix that would matter most: coarser, current-aware
sampling resolution (e.g. more, smaller pulses, or bins sized relative to
the actual per-pulse voltage step) would recover coverage without needing
to constrain the discharge rate itself.
