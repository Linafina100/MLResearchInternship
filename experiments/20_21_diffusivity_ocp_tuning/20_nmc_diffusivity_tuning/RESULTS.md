# Experiment 20: Closing the sim to real coverage gap, normalization failure, and an NMC diffusivity fix

This experiment contains three parts of the same investigation. They were originally run as separate experiments during development, but are collected here as one experiment so that the experiment numbering on `main` stays continuous.

## Part A: SOH scaling + `t_interp` fixes the coverage problem at SOH=0.8

### Motivation

Experiment 18 showed that `t_interp` can recover the missing synthetic NMC samples in the 2.5 to 3.0 V target zone for SOH >= 0.875.

However, at SOH=0.8, which is the important threshold for this project, Chen2020 and OKane2022 showed an almost instant drop in voltage.

The cause was traced to how SOH was implemented in the simulation.

The simulation scripts reduced the **Maximum concentration** in the positive and negative electrodes according to SOH. However, they did not reduce the **Initial concentration** at the same time.

This changes the ratio between the two concentrations. At lower SOH, the positive electrode can therefore be filled beyond its allowed stoichiometry range. This causes the sudden voltage collapse.

The fix is to scale both concentrations together. Since both values are changed by the same factor, their ratio stays the same.

This was verified directly: the electrode balance stays the same across different SOH values when both concentrations are scaled.

It was also tested in an actual discharge simulation. At SOH=0.8 with Chen2020 and OKane2022, there were no single step voltage jumps larger than 0.3 V after the fix. Before the fix, the simulation showed a voltage collapse of about 1.3 V.

### A mistake found during the experiment

The first version of the corrected simulation script only included the SOH scaling fix.

It still used PyBaMM's default adaptive solver:

```python
sim.solve(initial_soc=soc)
```

This gave **no improvement**. Accuracy stayed at 24.33%, the same as in experiments 16 and 17, and NMC recall remained 0.00.

The reason became clear when looking at the raw simulation output. Each battery still only had about 37 to 75 data points.

This was the separate sampling problem already identified in Experiment 18.

The SOH scaling fix removes the physical discontinuity in the simulation, but it does not make the default solver produce more data points in the target voltage range.

Therefore, both fixes are needed:

1. Scale `Initial concentration` together with `Maximum concentration`.
2. Use the dense `t_interp` method from Experiment 18.

Using only one of them is not enough.

### Method

The corrected version used:

* SOH fixed at 0.8
* C-rate randomized between 0.1 and 0.2C
* Temperature randomized between 15 and 35°C
* 1.5 V cutoff
* All three NMC parameter sets
* SOH scaling fix
* Dense `t_interp` output

All 498 simulations succeeded.

Each battery now contains about 3001 rows instead of the 37 to 75 rows produced by the default solver.

### Result: coverage is fixed

| Model         |   Accuracy |
| ------------- | ---------: |
| Random Forest | **27.81%** |
| XGBoost       |     25.82% |

The overall accuracy is still low, but NMC recall moved above zero for the first time in this investigation:

**0.00 → 0.04**

NMC precision was 0.98. This means that when the model predicted NMC, it was usually correct, but it still predicted NMC very rarely.

More importantly, the coverage problem was essentially solved.

| Voltage bin | Real LFP | Synthetic LFP | Real NMC | Synthetic NMC |
| ----------- | -------: | ------------: | -------: | ------------: |
| 3.0 to 2.9  |    99.5% |     **99.6%** |    99.4% |      **100%** |
| 2.9 to 2.8  |    99.5% |     **99.6%** |     100% |      **100%** |
| 2.8 to 2.7  |    99.5% |     **99.6%** |     100% |      **100%** |
| 2.7 to 2.6  |    99.5% |     **99.6%** |     100% |      **100%** |
| 2.6 to 2.5  |    99.5% |     **99.6%** |     100% |         73.5% |

This is a major improvement compared with experiments 16 and 17, where synthetic NMC coverage was only 0 to 1.6% in this region.

### A different problem becomes visible

Once the synthetic data finally had enough coverage, another problem became clear.

The magnitude of synthetic dV/dQ was much larger than in the real data.

| Voltage bin | Real LFP | Synthetic LFP | Real NMC | Synthetic NMC |
| ----------- | -------: | ------------: | -------: | ------------: |
| 3.0 to 2.9  |    -0.31 |         -1.88 |    -3.36 |         -9.90 |
| 2.9 to 2.8  |    -0.72 |         -2.84 |    -2.73 |        -13.63 |
| 2.8 to 2.7  |    -1.18 |         -4.64 |    -3.30 |        -17.62 |
| 2.7 to 2.6  |    -1.34 |         -6.67 |    -5.24 |        -26.22 |
| 2.6 to 2.5  |    -2.87 |         -8.72 |    -7.73 |        -16.17 |

Synthetic dV/dQ values are around 3 to 6 times larger than the real values.

This was not a new problem. It had already been seen in Experiment 07 and in the LFP results from experiments 16 and 17. The difference is that the coverage fix now makes it possible to clearly measure and investigate the problem.

### Part A conclusion

The combination of SOH scaling and dense `t_interp` sampling solves the coverage problem at SOH=0.8.

The remaining main problem is the difference in dV/dQ magnitude between the synthetic and real data.

This becomes the focus of Parts B and C.

The SOH scaling fix was not applied retroactively to earlier experiments. Those results were generated without this fix and should therefore be interpreted in that context.

---

## Part B: Per battery normalization does not solve the problem

### Motivation

Part A showed that synthetic and real dV/dQ values have very different magnitudes.

One possible solution was to remove the scale difference before training the ML models.

The idea was that if the synthetic and real batteries have similar **shape** but different overall magnitude, normalization could remove the scale difference and allow the model to learn the shape.

The existing simulations from Part A were reused. Only the ML preprocessing was changed.

### Method

Four normalization methods were tested:

* `none`: no normalization
* `l2`: scale each battery's feature vector to unit length
* `minmax`: scale each battery's five values to the range 0 to 1
* `standardize`: calculate a z score separately for each battery

The rest of the ML pipeline was unchanged.

### Result: raw accuracy was misleading

| Variant     | Model | Raw accuracy | **Balanced accuracy** | LFP recall | NMC recall |
| ----------- | ----- | -----------: | --------------------: | ---------: | ---------: |
| none        | RF    |       27.81% |                52.07% |      0.997 |      0.045 |
| none        | XGB   |       25.82% |                50.87% |      1.000 |      0.017 |
| l2          | RF    |   **73.17%** |                50.15% |      0.050 |      0.953 |
| l2          | XGB   |   **72.74%** |                50.04% |      0.055 |      0.946 |
| minmax      | RF    |       62.06% |                52.14% |      0.327 |      0.716 |
| minmax      | XGB   |   **75.46%** |                50.32% |      0.010 |      0.996 |
| standardize | RF    |       52.69% |                35.42% |      0.016 |      0.693 |
| standardize | XGB   |        2.63% |                 2.15% |      0.012 |      0.031 |

At first glance, some of the raw accuracy values look much better. For example, L2 normalization gives about 73% accuracy and minmax + XGBoost gives 75%.

However, these numbers do not represent real classification improvement.

The test set contains:

* 578 LFP
* 1781 NMC
* 2359 total
* **75.50% NMC**

Therefore, a model that simply predicts NMC for every battery would already get 75.50% raw accuracy.

This is exactly what happened with some of the normalization methods.

For example, L2 normalization caused the models to predict NMC for most samples. The raw accuracy therefore increased because NMC is the majority class.

The balanced accuracy stayed around 50%, meaning that the model was not actually learning to distinguish the two chemistries.

### Why the normalization did not work

The mismatch between synthetic and real data is not just one constant scaling factor.

For example, for NMC:

* 2.7 to 2.6 V: synthetic values are about 5 times larger
* 2.6 to 2.5 V: synthetic values are about 2.1 times larger

The difference therefore changes between voltage bins.

This means that the synthetic and real curves are not simply the same shape multiplied by one constant.

Removing the overall scale from each battery therefore does not recover the same underlying shape.

### Part B conclusion

Per battery normalization does not solve the sim to real problem.

It mainly changes which class the model tends to predict when it fails.

This experiment also shows why **balanced accuracy and per class recall are important when the test set is imbalanced**.

Raw accuracy alone can make a model look much better even when it has almost no ability to distinguish the two classes.

The magnitude mismatch therefore remained unresolved going into Part C.

---

## Part C: Changing NMC diffusivity to reduce the magnitude mismatch

### Motivation

Since normalization did not solve the magnitude problem, the next approach was to change the simulation itself.

The goal was to make the simulated NMC discharge curve less steep and therefore more similar to the real data.

The first step was to determine which physical parameter actually controls this steepness.

### Which parameter affects the curve?

Three possible mechanisms were tested using one controlled battery:

* Chen2020
* SOH=0.8
* 0.15C
* resistance_factor=0.8
* dense `t_interp` output

| Parameter                                            | What it affects                        | Result           |
| ---------------------------------------------------- | -------------------------------------- | ---------------- |
| Positive electrode conductivity / contact resistance | Ohmic voltage drop                     | **No effect**    |
| Positive electrode exchange current density          | Reaction rate                          | **Small effect** |
| **Positive particle diffusivity**                    | Lithium diffusion inside NMC particles | **Clear effect** |

Reducing the positive particle diffusivity from the baseline eventually changed the mean magnitude from about -8.02 to -5.92.

### Why diffusivity works

Resistance mainly changes the voltage by:

**Voltage drop = current × resistance**

During approximately constant current discharge, this value stays relatively constant. It therefore shifts the voltage curve rather than changing its shape.

Because dV/dQ describes how quickly voltage changes with capacity, simply shifting the whole curve does not solve the problem.

Diffusivity works differently.

It controls how quickly lithium can move inside the NMC particle. The particle surface concentration affects the OCP, while the bulk concentration represents the lithium remaining in the particle.

Changing diffusivity therefore changes how quickly the surface concentration follows the bulk concentration.

This changes the **shape and steepness** of the discharge curve, which is what is needed here.

### Finding a safe diffusivity range

A first test showed that the useful range was fairly narrow.

* 10 times lower diffusivity: good result
* 15 times lower: only one extreme point remained
* 20 times lower or more: target zone disappeared

This showed that reducing diffusivity too much can cause the same type of simulation collapse seen earlier, although the cause is now the diffusivity change rather than the SOH scaling problem.

## Phase 1: Systematic diffusivity test

`diagnose_diffusivity_factor.py` tested the following reduction factors:

**1, 4, 6, 8, 10 and 12**

The test covered:

* All three NMC parameter sets
* C-rates of 0.1, 0.15 and 0.2C
* Temperatures of 15, 25 and 35°C
* SOH=0.8
* Dense `t_interp` output

One implementation issue was also found during this process.

Chen2020 defines diffusivity as a constant, while Mohtat2020 and OKane2022 define it as a function of stoichiometry and temperature.

The code was therefore changed so that the callable function was wrapped before applying the reduction factor.

### Results

Two of the three parameter sets responded clearly to the diffusivity change.

| Parameter set | Factor 1                   | Factor 10                     |
| ------------- | -------------------------- | ----------------------------- |
| Chen2020      | Mean -8.04, 3/9 in target  | **Mean -5.81, 9/9 in target** |
| OKane2022     | Mean -9.71, 0/9 in target  | **Mean -6.27, 9/9 in target** |
| Mohtat2020    | Mean -27.76, 0/9 in target | Mean -26.45, 0/9 in target    |

For Chen2020 and OKane2022, a factor of **10** gave a good and stable result across the tested C-rates and temperatures.

A factor of 12 was already less stable. Both parameter sets had only 6/9 successful target-zone combinations at that setting.

Mohtat2020 behaved differently.

Its mean magnitude only changed from -27.76 to -26.45 across the tested factors. It also continued to produce the same nine outlier combinations.

This is consistent with the different behaviour of Mohtat2020 already seen in experiments 07 and 18.

The results therefore suggest that diffusivity is not the main parameter controlling the curve for Mohtat2020.

### Decision about Mohtat2020

Mohtat2020 was removed from the NMC parameter set used in the next phase.

This does not mean that Mohtat2020 is solved. It remains a separate open problem.

The decision was made so that the next phase could focus on the two parameter sets where the diffusivity change clearly worked.

Full diagnostic results are stored in `diagnostic_results.csv`.

## Phase 2: Full pipeline with diffusivity/10

The final simulation used:

* SOH=0.8
* C-rate between 0.1 and 0.2C
* 1.5 V cutoff
* Dense `t_interp` output
* Chen2020 + OKane2022
* Mohtat2020 excluded
* Positive particle diffusivity divided by 10

All 498 simulations succeeded.

### Result: Random Forest shows a clear sim to real signal

| Model             | Raw accuracy | **Balanced accuracy** | LFP recall | NMC recall |
| ----------------- | -----------: | --------------------: | ---------: | ---------: |
| **Random Forest** |       74.65% |            **82.92%** |      0.991 |      0.667 |
| XGBoost           |       28.78% |                52.60% |      0.993 |      0.059 |

The Random Forest result is important because its balanced accuracy is 82.92%, well above the 50% chance level.

Both classes also have reasonably high recall:

* LFP: 0.991
* NMC: 0.667

This is different from the normalization results in Part B, where one class was usually predicted almost exclusively.

For the first time in this coverage and magnitude investigation, the synthetic data produced a clear two way classification signal in the real data.

XGBoost still did not show the same improvement.

Its balanced accuracy was 52.60%, which is close to chance, and NMC recall was only 0.059.

This means that XGBoost still predicted LFP for most samples.

The different behaviour between Random Forest and XGBoost is similar to a pattern seen earlier in Experiment 07. Random Forest averages many trees, which can make it less sensitive to differences between the training and test distributions. XGBoost builds trees sequentially and can therefore be more sensitive to the exact value ranges in the training data.

### Why the diffusivity fix is not a complete solution

The magnitude match improved, but not equally across the entire target zone.

| Voltage bin | Real NMC | Synthetic NMC before fix | Synthetic NMC after fix | Real LFP | Synthetic LFP |
| ----------- | -------: | -----------------------: | ----------------------: | -------: | ------------: |
| 3.0 to 2.9  |    -3.36 |                    -9.90 |               **-3.57** |    -0.31 |         -1.88 |
| 2.9 to 2.8  |    -2.73 |                   -13.63 |                   -5.95 |    -0.72 |         -2.83 |
| 2.8 to 2.7  |    -3.30 |                   -17.62 |                  -10.74 |    -1.18 |         -4.65 |
| 2.7 to 2.6  |    -5.24 |                   -26.22 |                  -14.51 |    -1.34 |         -6.66 |
| 2.6 to 2.5  |    -7.73 |                   -16.17 |                  -16.74 |    -2.87 |         -8.72 |

The improvement is strongest at the top of the target zone.

For 3.0 to 2.9 V:

**Real: -3.36**

**Synthetic: -3.57**

This is a very close match.

Further down the voltage range, the difference becomes larger again.

For example, in the 2.7 to 2.6 V bin:

**Real: -5.24**

**Synthetic: -14.51**

The deepest bins therefore remain much too large.

NMC coverage also becomes slightly lower in the deepest bins:

* 2.7 to 2.6 V: 89.2%
* 2.6 to 2.5 V: 83.1%

The diffusivity change therefore improves the magnitude match higher in the target zone, but comes with a small loss of coverage at the bottom.

LFP was not changed in this experiment. Its separate magnitude mismatch remains.

## Overall conclusion

The three parts of Experiment 20 address three different issues.

### Part A: Coverage

The combination of SOH scaling and dense `t_interp` sampling solves the missing synthetic data problem at the realistic SOH=0.8 threshold.

### Part B: Normalization

Per battery normalization does not solve the difference between synthetic and real dV/dQ.

The apparently high raw accuracy from some normalization methods was caused by the imbalanced test set. Balanced accuracy and per class recall showed that the models were mostly switching between predicting LFP or NMC rather than learning a real distinction.

### Part C: NMC diffusivity

Reducing the positive particle diffusivity by a factor of 10 improves the NMC magnitude substantially for Chen2020 and OKane2022.

Random Forest reached **82.92% balanced accuracy**, which is the first clear sim to real classification signal found in this investigation.

However, the improvement is not uniform across the entire target zone, and XGBoost still performs close to chance.

LFP's separate magnitude mismatch is also still unresolved.

### Next steps

Two main questions remain:

1. **Why does XGBoost respond differently from Random Forest?**
   The same improved features are given to both models, but only Random Forest shows a clear improvement. It may therefore be useful to investigate whether Random Forest is better suited to this feature set.

2. **Can the remaining NMC magnitude mismatch be reduced?**
   A single diffusivity factor affects the whole positive electrode in the same way. A more detailed adjustment, for example one that changes diffusivity depending on the region of the particle or operating condition, could potentially improve the deeper voltage bins.

The separate LFP mismatch could also be investigated using the same general approach, but this was not part of Experiment 20.

Full run outputs:

`sim_and_real_raw.csv`

`features/ml_features.csv`

The feature file can be regenerated using `simulate_batteries_diffusivity_tuned.py` followed by `evaluate_diffusivity_tuned_sim_to_real.py`.
