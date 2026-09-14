# Bin-interpolation fix for the sampling-resolution problem

## Motivation

`experiments/04_pulse_variable_discharge_soc_sweep/RESULTS.md`'s root-cause
section found that randomizing pulse discharge rate collapses signal in 3
of 4 SOC intervals because higher current makes each pulse's voltage jump
bigger, so consecutive relaxed samples increasingly land on either side of
a narrow 0.1V bin instead of inside it. The proposed fix: instead of
assigning a pulse's averaged dV/dQ to only the bin containing its ending
voltage, assign that same value to **every** bin the pulse's voltage span
crosses.

## Implementation

`feature_engineering_interpolated.py` (this folder) is a modified copy of
`feature_engineering.py`: for each valid pulse transition, it computes the
bin range between the starting and ending voltage (`_voltage_to_bin_high`
applied to both ends) and writes the pulse's one averaged `dV/dQ` value
into every bin in that range, not just the destination bin. A small pulse
that stays within one bin behaves identically to the original code (this
is a strict superset of the original single-bin logic). Verified directly:
for a synthetic 3.55V -> 3.32V transition, the fix correctly identifies
bins `[3.4, 3.5, 3.6]` as spanned, matching the worked example in the
fix's design explanation.

No simulation changes needed -- `reevaluate_with_interpolation.py` reuses
the exact raw CSVs experiment 04 already generated
(`data/pulse_variable_soc_<interval>/raw/`), re-running only feature
engineering (interpolated) and `ml_pipeline.py` (unchanged, from the repo
root) against them.

## Results: before vs. after interpolation

| SOC Interval | Before (RF/XGB) | After (RF/XGB) | Bin(s), both cases |
|---|---|---|---|
| 0.7-1.0 | 81.61% / 81.61% | **74.71% / 75.86%** | `dV_dQ_V_3.4_3.3` |
| 0.5-0.8 | none / none | **none / none** (unchanged) | — |
| 0.3-0.6 | none / none | **none / none** (unchanged) | — |
| 0.1-0.4 | 94.85% / 91.75%\* | **68.29% / 70.73%** | `dV_dQ_V_3.3_3.2` |

\*0.1-0.4's "before" number is from the original *fixed-rate* pulse sweep
(`experiments/01_.../soc_sweep.py`), the closest prior comparison since
this exact interval wasn't part of the variable-rate sweep's headline
table at this specific bin — included for context, not a strict
apples-to-apples baseline like the other three rows.

**The fix does not help, and actively hurts where it used to work.**

## Why it didn't rescue the two failed intervals

Checking `dV_dQ_V_3.4_3.3`'s coverage directly in the 0.5-0.8 interval:
LFP coverage is **exactly 9.5% before and after interpolation** --
completely unchanged, while *other*, lower-voltage bins' coverage rose
substantially with interpolation (e.g. LFP's `3.0-2.9` bin: 13.2% ->
36.2%, `2.9-2.8`: 7.0% -> 23.5%). This proves the interpolation mechanism
itself works correctly -- it's genuinely filling in previously-skipped
bins. But for the *specific* bin that determines classification here, it
found no bracketing pair of samples to interpolate across at all: many
LFP cells in this SOC window simply never occupy the 3.4-3.3V region
during the entire test, at any sampling resolution, because their voltage
trajectory (set by `Initial_SOC`, independent of pulse size) starts and
stays below it. Interpolation only fills gaps *between* real samples --
it can't manufacture a data point in a region a battery's voltage never
visited. And NMC's coverage in the *other* now-improved bins (2.6-3.1V)
stays at 0% throughout -- NMC's chemistry just doesn't reach that low in
this SOC window -- so the `>=20%` mutual-coverage filter still has nothing
to work with there either. **This confirms the two collapse mechanisms
diagnosed earlier are genuinely independent: sampling resolution (which
this fix addresses) and SOC-driven voltage-range non-overlap (which it
doesn't, and can't).**

## Why it hurts the two intervals that used to work

More surprising: accuracy *drops* substantially in both intervals that
still have a surviving bin (0.7-1.0: -6 to -7 points; 0.1-0.4: -21 to -26
points), even though the *same* bin survives the coverage filter before
and after. The likely explanation: interpolation fills a bin with the
*whole pulse's* averaged slope, smeared across every bin that pulse
crossed -- for large, high-current pulses this is a coarse approximation,
not a locally accurate measurement of what the true slope was specifically
within that narrow window. Previously, a battery with no real transition
landing in a bin was left `NaN` and median-imputed to a single constant
(itself a crude fallback, but at least a *consistent* one); now it often
gets a genuine-but-approximate borrowed value instead, which adds real
variance without adding real local signal -- diluting what was otherwise
a cleanly-measured, discriminative feature for the batteries that *did*
land there normally.

## Conclusion

The interpolation fix is **not a net improvement** for this dataset. It
correctly patches the diagnosed sampling-resolution mechanism (verified:
coverage genuinely rises in bins that were being skipped), but:

1. It cannot rescue intervals whose real constraint is SOC-driven
   voltage-range non-overlap between chemistries, not sampling
   resolution -- these are independent problems requiring independent
   fixes.
2. It measurably degrades accuracy in intervals that already had adequate
   resolution, by replacing missing-but-consistently-imputed values with
   present-but-approximate ones.

This suggests bin interpolation, as implemented, is the wrong tool here.
The two alternatives discussed but not implemented earlier (widening bins
uniformly, or making pulse duration scale with C-rate to keep sampling
density itself constant) remain open -- the latter in particular would
fix the resolution problem at its source instead of patching around it
after the fact, and wouldn't introduce the smearing/dilution effect this
approach did.
