# UU pipeline, experiments 02-04: chasing experiment 01's NMC_recall=0.0

Consolidated writeup for three sequential experiments. Each fixes a real
problem the previous one exposed; none alone produces a trustworthy
NMC-vs-LFP number, but together they narrow down where the actual ceiling
is.

## The question this answers

Experiment 01 (`01_simulated_soh_match`) reported `NMC_recall == 0.0` for
every `Initial_SOC` bucket from 1.0 down to 0.2, flipping to 0.73-0.92 at
the three lowest buckets. Is that a real sim-to-real failure, or an
artifact of how the real NMC `.xlsx` files get parsed?

## Part A -- experiment 02: contiguous-segment fix + a second, independent bug

**Root cause #1 (real-data parsing).** `build_UU_dataset.py`'s
`process_nmc_file()` grouped raw rows by `(Cycle Index, Step Index)` and
treated each group as one continuous discharge trace. In the raw `.xlsx`
files, that same label recurs 100+ times non-contiguously throughout the
file (a GITT/pulse-style protocol, not a single discharge) -- each
recurrence resets its own local `Time` column to `00:00:00`. Merging and
sorting by that local time interleaves dozens of separate ~5-minute pulses
into one scrambled trace, which reintegrates to a near-zero, near-identical
"capacity" for every cell (verified: mean 0.0001133 Ah, std 6e-9 across 198
independent real NMC cells -- the smoking gun that this was an artifact,
not measured capacity). That degenerate trace's `dQ` then falls below
`feature_engineering.py`'s `dQ > 1e-5` noise filter, silently dropping
every NMC battery above `Initial_SOC=0.2`.

Fix: split each `(Cycle Index, Step Index)` group into contiguous runs
using `DataPoint` (a true monotonic row index) before computing capacity.
Confirmed systemic across all three NMC files checked (`00467`, `88803`,
`04809`): ~100-120 non-contiguous repeats per discharge step, every time.

**Root cause #2 (found while verifying #1, independent bug).**
`run_evaluation.py` computed `Battery_ID` via
`raw_df.groupby(GROUPBY_COLS).ngroup()` (default `sort=True`), while
`feature_engineering.py`'s own internal `Battery_ID` uses
`df.groupby(groupby_cols, sort=False).ngroup()`. These assign **different
integer labels to the same groups** -- verified every single one of 10.5M
rows differs between the two calls. Since `run_evaluation.py` used its own
(wrong) IDs to map `DataKind`/`Initial_SOC`/`Variation_ID` back onto the
feature table, every row's SOC-bucket and real/synthetic label was
silently cross-wired onto some other battery's features. (This is how
experiment 01's "real NMC survives at low SOC" numbers appeared --
verified the raw data has zero real NMC rows at `Initial_SOC=0.05`.) Fixed
by adding `sort=False` to match.

**Result after both fixes:** the honest picture is worse than illusory
high numbers -- it's "no real NMC signal at all, and real LFP is 100%
misclassified."

| | real LFP | real NMC |
|---|---|---|
| surviving batteries (any SOC bucket) | 8,428 | **0** |

Every real NMC segment's true capacity caps out around 0.000113 Ah
(confirmed: no segment across the whole dataset exceeds this) -- about
20,000x smaller than the ~2 Ah scale synthetic data and the `dQ > 1e-5`
filter assume, so it never clears that absolute threshold at any
truncation depth. Real LFP (whose segments run ~0.0014 Ah, still far below
2 Ah) gets 0% recall because its `dV/dQ` magnitude in absolute Ah is
100-250x larger than *either* synthetic class (e.g. median -3634 vs.
synthetic LFP's -39 and synthetic NMC's -24 at one bin) -- clipping to the
training distribution's 1st-99th percentile pushes it to whichever edge
happens to be NMC.

Root tie-together: **real UU capacity (both chemistries) was never
rescaled to the synthetic data's ~2 Ah scale**, despite a
`Target_Capacity_Ah=2.0` field that implies that was the intent -- it's
stored as metadata but never applied anywhere.

## Part B -- experiment 03: SOC-fraction (capacity) normalization

**Change:** forked `feature_engineering.py` to compute `dV/d(SOC fraction)`
per transition instead of `dV/dQ` in absolute Ah, where SOC fraction =
`(Q - Q_min) / (Q_max - Q_min)` using each battery's own realized capacity
span. Makes a tiny real segment and a full-size synthetic cell comparable
regardless of absolute scale. Also switched the noise filter to a relative
`dQ_frac > 1e-6` threshold (scale-invariant version of the same check).

**Result:** fixed the scale problem on its own terms -- real NMC: 36,429
segments now pass the per-transition filter (up from 0); real LFP: 22,033.
But every one of the 14 voltage bins got dropped by the (still
absolute-voltage, still pooled-across-SOC) `>=20% mutual coverage` column
filter:

| Voltage bin | LFP coverage | NMC coverage |
|---|---|---|
| 2.6-3.3V (LFP's plateau) | 60-93% | 6-15% |
| 3.4-4.1V (NMC's slope) | 0-3% | 9-20% |

No bin ever has both chemistries >=20% present. Re-ran the coverage check
per-`Initial_SOC`-bucket instead of pooled across all 12 -- same outcome,
because the real constraint isn't pooling, it's that LFP and NMC occupy
almost disjoint absolute voltage windows in this data (LFP's plateau vs.
NMC's higher-voltage pulses), so "both chemistries present in this bin" is
never satisfiable regardless of how coverage is grouped.

## Part C -- experiment 04: voltage-fraction normalization

**Change:** forked `feature_engineering.py` again to bin by
`(V - V_min) / (V_max - V_min)` (each battery's own observed voltage span)
instead of absolute volts, and dropped absolute voltage as a feature
entirely (explicit choice, not a safety-preserving compromise -- see
tradeoff below). Every battery then spans the same normalized 0-1 range by
construction, independent of its absolute voltage window.

**Tradeoff, stated plainly:** absolute voltage *range* is one of the most
physically legitimate LFP-vs-NMC signals there is (LFP tops out ~3.65V,
NMC reaches ~4.2-4.8V). Normalizing it away discards that signal entirely,
leaving only curve *shape* (how `dV/d(SOC fraction)` varies across the
normalized voltage-position axis) to distinguish the two chemistries.

**Result:** all 10 bins now survive (coverage no longer structurally
excludes either chemistry -- confirms the Part B diagnosis). But the real
signal is weak:

| Initial_SOC | n | RF balanced | LFP recall | NMC recall |
|---|---|---|---|---|
| 1.0 | 5,766 | 54.69% | 0.428 | 0.666 |
| 0.9 | 5,764 | 22.47% | 0.449 | 0.000 |
| 0.8 | 5,762 | 32.09% | 0.642 | 0.000 |
| 0.7 | 5,760 | 43.23% | 0.865 | 0.000 |
| 0.6 | 5,758 | 47.34% | 0.947 | 0.000 |
| 0.5 | 5,753 | 49.18% | 0.984 | 0.000 |
| 0.4 | 5,746 | 49.78% | 0.996 | 0.000 |
| 0.3 | 5,734 | 49.51% | 0.990 | 0.000 |
| 0.2 | 5,700 | 48.91% | 0.978 | 0.000 |
| 0.15 | 2,883 | 48.31% | 0.966 | 0.000 |
| 0.1 | 2,020 | 48.55% | 0.971 | 0.000 |
| 0.05 | 1,816 | 99.01% | 0.990 | n/a (0 real NMC survives this bucket) |

NMC recall is 0.0 in every bucket except `Initial_SOC=1.0` (the one bucket
using the *untruncated* full segment, where it's 0.666 but LFP recall drops
to 0.428 -- net balanced accuracy only 54.69%). Every other bucket sits
near chance level (22-50%).

## Interpretation

With the segmentation bug, the `Battery_ID` bookkeeping bug, the absolute
capacity scale mismatch, and the absolute voltage-window mismatch all
fixed, the ceiling on this real NMC data's learnable signal is still low.
That's informative rather than a dead end: it's consistent with the real
NMC segments being tiny, low-current (~0.27-1.36 mA) GITT-style
characterization pulses -- not genuine discharge curves -- whose *shape*
doesn't reliably resemble a simulated full discharge's shape even after
every absolute-scale confound is removed. Further filter engineering on
top of this pipeline is unlikely to fix that; the open question is whether
these NMC `.xlsx` files contain a genuine deep discharge anywhere at all
(none found in cycles 1-5 of any file checked), or whether a different
real NMC source is needed for this classification task.

## Files

- `UU_pipeline/_experiment/02_nmc_contiguous_segment_fix/data_generation/build_UU_dataset.py`
  -- NMC contiguous-segment fix (Part A, bug #1). `process_lfp_file()`
  unchanged.
- `UU_pipeline/_experiment/02_nmc_contiguous_segment_fix/run_evaluation.py`
  -- `Battery_ID` `sort=False` fix (Part A, bug #2); also persists
  `n_lfp`/`n_nmc` per bucket into `sweep_results.json` (experiment 01 only
  printed this to stdout, which is exactly what hid bug #2).
- `UU_pipeline/_experiment/03_soc_normalized_features/pipeline/feature_engineering.py`
  -- SOC-fraction (capacity) normalization (Part B); per-SOC-bucket mutual
  coverage check.
- `UU_pipeline/_experiment/04_voltage_normalized_features/pipeline/feature_engineering.py`
  -- adds voltage-fraction normalization on top of Part B (Part C).
- `outputs/sweep_results.json`, `outputs/plots/` in each experiment's own
  folder.

## Next steps

Not pursued here: re-adding absolute voltage as a side feature alongside
the normalized shape features (would recover the voltage-range signal but
reintroduce the Part B coverage problem for that feature specifically --
needs its own, separate handling, e.g. as unbinned per-battery scalar
features rather than binned columns); auditing `process_lfp_file()` for an
analogous segment-reuse artifact (not found yet, but not specifically ruled
out either); checking whether any UU NMC file contains a real deep
discharge outside cycles 1-5.

This chain also left experiment 01's original `run_evaluation.py` with the
same `Battery_ID` `sort=True` bug (Part A, bug #2) uncorrected -- its
sweep_results.json numbers should be treated as unreliable until that's
fixed there too.
