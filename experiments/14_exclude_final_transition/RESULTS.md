# Excluding each battery's final transition instead of whole bins

## What this experiment is

Two prior attempts at fixing the termination-step artifact (experiment 11's
discovery, experiment 12's quantification): experiment 12 excludes any
feature bin within one bin-width of either chemistry's cutoff -- coarse
but effective. Experiment 13 tried fixing it at the simulation level
(finer PyBaMM output resolution) and found that makes the artifact worse,
not better -- ruled out.

This experiment tries a more surgical alternative: root
`feature_engineering.py` gained an `exclude_final_transition` flag that
drops only each battery's single *final* raw dV/dQ transition before
binning -- the one transition experiment 11 showed is always an
irregular, oversized step landing right on the voltage cutoff -- rather
than dropping whole bins after the fact. The hope was that the existing
`>=20%` mutual-coverage filter would then reject exactly the bins that
were only ever populated by the artifact, on its own, without needing to
know where the cutoffs are.

Same reevaluate-existing-data pattern as experiments 05 and 12: reuses
experiment 03's already-committed raw CSVs, no re-simulation.

## Results

| SOC Interval | Bin Set | RF Accuracy | XGB Accuracy | Surviving Bins |
|---|---|---|---|---|
| 0.7-1.0 | before | 73.96% | 78.12% | `3.3-3.2` |
| 0.7-1.0 | after | 73.96% | 78.12% | `3.3-3.2` (unchanged) |
| 0.5-0.8 | before | 98.96% | 98.96% | `3.3-3.2`, `3.2-3.1` |
| 0.5-0.8 | after | 98.96% | 98.96% | `3.3-3.2`, `3.2-3.1` (unchanged) |
| 0.3-0.6 | before | 98.96% | 98.96% | `3.3-3.2`, `3.2-3.1`, `2.4-2.3`, `2.3-2.2` |
| 0.3-0.6 | after | **100.00%** | 98.96% | `3.3-3.2`, `3.2-3.1` |
| 0.1-0.4 | before | 94.74% | 94.74% | `3.2-3.1`, `2.4-2.3`, `2.3-2.2` |
| 0.1-0.4 | after | **73.68%** | **75.79%** | `3.2-3.1` |

**Identical to experiment 12's results at every interval, to the decimal**
-- same surviving bins, same accuracy, same feature importances. Dropping
just the final transition per battery is enough to make `2.4-2.3` and
`2.3-2.2` fail the mutual-coverage filter on their own at SOC 0.3-0.6 and
0.1-0.4 (NMC's coverage in those bins drops to 0% once the artifact
transition is gone -- confirming, as suspected, that NMC's entire presence
there *was* the artifact, with zero genuine mid-trace signal underneath
it to preserve). No bin that had genuine cross-chemistry coverage was
affected.

## Why this is the better fix to keep, despite the identical numbers

Both approaches reach the same answer here, but for different reasons:

- Experiment 12 (bin exclusion) needs to be told the cutoff voltages and
  a safety margin ("one bin-width"), hardcoded per experiment. It would
  need re-tuning for experiment 06's different cutoffs (1.5V/1.8V), or
  any future cutoff choice, and risks either under- or over-excluding if
  the margin doesn't match the actual jump size.
- Experiment 14 (this one, transition exclusion) needs no knowledge of
  cutoff values at all. It removes the one raw sample per battery that is
  structurally always suspect (PyBaMM's event-triggered last step,
  regardless of what voltage that happens to be at), and lets the
  pipeline's own existing coverage filter make the bin-level call from
  the resulting, artifact-free data -- the filter doing exactly the job
  it was already designed to do, just no longer fed contaminated input.

Practically: since both give the same answer on this dataset, either is a
valid fix for experiment 03 specifically. `exclude_final_transition` is
the one worth generalizing to other experiments/cutoff choices, since it
doesn't need per-experiment recalibration.

## Not yet done

- Only exercised against experiment 03's data (same 4 SOC intervals). Not
  re-run against experiment 06's data (different cutoffs, randomized
  C-rate) to confirm it behaves sensibly there too -- expected to be a
  no-op there per experiment 11's inference that 06's artifact bins never
  survive coverage in the first place, but not directly re-verified with
  this flag.
- `exclude_final_transition` defaults to `False` in root
  `feature_engineering.py`, so no existing caller's behavior changed by
  adding it. Whether to flip the default (or have every sweep script pass
  `True`) once this is trusted more broadly is a separate decision, not
  made here.
- Experiment 03's own `RESULTS.md` still documents the uncorrected,
  artifact-inflated 0.1-0.4 number and is unedited by this experiment.
