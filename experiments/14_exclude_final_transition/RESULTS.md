# Excluding Each Battery's Final Transition Instead of Whole Bins

## What this experiment tests

Previous experiments found that the final discharge step creates an artifact in the dV/dQ features. Experiment 12 addressed this by removing entire feature bins close to the voltage cutoff. Experiment 13 tested whether using a finer PyBaMM output resolution would remove the artifact, but instead found that it made the problem worse.

This experiment tests a more targeted solution. Instead of removing entire bins, the feature engineering step now has an `exclude_final_transition` option that removes only the final dV/dQ transition from each battery before binning. This is the transition that was identified in experiment 11 as an unusually large step caused by PyBaMM reaching the voltage cutoff.

The idea is that, after removing this artifact, the existing `>=20%` mutual coverage filter should automatically remove any bins that no longer contain enough data from both chemistries. This avoids having to manually specify which bins are affected by the cutoff.

As in experiments 05 and 12, this experiment reuses the existing raw CSV files from experiment 03 rather than running new simulations.

## Results

| SOC Interval | Bin Set | RF Accuracy | XGB Accuracy | Surviving Bins                             |
| ------------ | ------- | ----------: | -----------: | ------------------------------------------ |
| 0.7–1.0      | Before  |      73.96% |       78.12% | `3.3–3.2`                                  |
| 0.7–1.0      | After   |      73.96% |       78.12% | `3.3–3.2`                                  |
| 0.5–0.8      | Before  |      98.96% |       98.96% | `3.3–3.2`, `3.2–3.1`                       |
| 0.5–0.8      | After   |      98.96% |       98.96% | `3.3–3.2`, `3.2–3.1`                       |
| 0.3–0.6      | Before  |      98.96% |       98.96% | `3.3–3.2`, `3.2–3.1`, `2.4–2.3`, `2.3–2.2` |
| 0.3–0.6      | After   | **100.00%** |       98.96% | `3.3–3.2`, `3.2–3.1`                       |
| 0.1–0.4      | Before  |      94.74% |       94.74% | `3.2–3.1`, `2.4–2.3`, `2.3–2.2`            |
| 0.1–0.4      | After   |  **73.68%** |   **75.79%** | `3.2–3.1`                                  |

The results are identical to experiment 12 at every SOC interval, including the surviving bins, model accuracy, and feature importances.

Removing the final transition causes the `2.4–2.3` and `2.3–2.2` bins to fail the mutual coverage requirement at SOC 0.3–0.6 and 0.1–0.4. After removing the artifact, NMC has no remaining data in these bins. This confirms that the NMC signal in these bins came entirely from the final transition artifact.

Importantly, bins containing genuine data from both chemistries were not affected.

## Why this is the better fix

Both experiment 12 and this experiment lead to the same results, but the transition based approach is more general.

Experiment 12 requires the cutoff voltage and a safety margin to be specified manually. This would need to be adjusted for experiments with different cutoff voltages and could result in removing either too much or too little data.

The new `exclude_final_transition` approach does not need to know the cutoff voltage. It simply removes the final transition from each battery, which is the part of the data known to be affected by the PyBaMM cutoff event. The existing coverage filter then determines which bins still contain enough valid data.

Since both approaches give the same results for experiment 03, both are valid fixes for this dataset. However, `exclude_final_transition` is more suitable for use across different experiments because it does not depend on specific cutoff voltages.

## Not yet tested

* The new option has only been tested on experiment 03. It has not yet been tested on experiment 06, which uses different cutoff voltages and randomized C rates.
* `exclude_final_transition` is set to `False` by default, so existing experiments are not affected unless the option is explicitly enabled.
* Experiment 03's `RESULTS.md` still contains the original results, including the artifact affected 0.1–0.4 result, and has not been updated by this experiment.
