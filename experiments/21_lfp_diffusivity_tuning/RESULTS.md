# Experiment 21: Does diffusivity tuning also fix the LFP magnitude mismatch?

## Motivation

In Experiment 20, changing the diffusivity of NMC particles reduced most of the difference in dV/dQ magnitude between the synthetic and real data. This was the first clear sign that the simulations could be made more similar to the real data.

LFP has a similar problem, but the difference is even larger. Its synthetic dV/dQ values are typically 2 to 6 times larger than the real values. This problem has been seen since Experiment 07 but had not yet been investigated.

The goal of this experiment was therefore to test whether the same approach used for NMC could also fix the LFP mismatch.

## Phase 1: Testing diffusivity tuning

First, the same approach as for NMC was tested on LFP.

At one test point (Prada2013, SOH=0.8, 0.15C, resistance_factor=0.8), the LFP positive particle diffusivity was changed over a very large range. The baseline value is 5.9e-18 m²/s, which is already much lower than the NMC value of 4e-15 m²/s.

Changing the LFP diffusivity by factors from 1/50 to 10 had essentially no effect. The mean dV/dQ magnitude stayed at about -3.38 throughout.

Two other parameters were also tested:

* Positive electrode exchange current density: no measurable effect.
* Negative electrode diffusivity: changing it moved the magnitude in the wrong direction.

### Systematic test

The diffusivity test was then repeated more systematically using `diagnose_lfp_diffusivity_factor.py`.

Factors of 1, 5, 10, 20 and 50 were tested across:

* C-rates: 0.1, 0.15 and 0.2C
* Temperatures: 15, 25 and 35°C
* SOH: 0.8

This gave 45 simulations in total. All simulations were successful.

The result was very consistent: the mean magnitude stayed between -3.36 and -3.37 for every combination. The real LFP target range is -2.9 to -0.3, so none of the simulations reached the target or moved meaningfully closer to it.

| Factor       | Successful | In real target (-2.9 to -0.3) | Mean magnitude |
| ------------ | ---------: | ----------------------------: | -------------: |
| 1 (baseline) |        9/9 |                           0/9 |          -3.37 |
| 5            |        9/9 |                           0/9 |          -3.37 |
| 10           |        9/9 |                           0/9 |          -3.37 |
| 20           |        9/9 |                           0/9 |          -3.37 |
| 50           |        9/9 |                           0/9 |          -3.36 |

### Why this is different from NMC

The reason diffusivity worked for NMC but not for LFP is likely related to how the two materials behave during discharge.

NMC has a relatively smooth voltage curve. Near the end of discharge, the voltage depends strongly on how quickly lithium can move through the particle. This is affected by solid-state diffusivity, so changing diffusivity can change the shape and magnitude of dV/dQ.

LFP behaves differently. LiFePO4 undergoes a two-phase transition during charging and discharging. This creates the characteristic flat voltage plateau.

The part of the curve we are studying here, around 2.5 to 3.0 V, is mainly controlled by the fitted OCP curve. In other words, the shape of the OCP function has a larger effect than transport parameters such as diffusivity.

This explains why changing diffusivity, exchange current density and the other electrode's diffusivity did not solve the problem. These parameters do not change the shape of the OCP function itself.

### Phase 1 conclusion

The diffusivity approach that worked for NMC does not work for LFP.

Therefore, there is no reason to continue testing LFP diffusivity, exchange current density or negative electrode diffusivity for this particular problem unless new evidence suggests otherwise.

Full results are stored in `diagnostic_results.csv`.

## Phase 2: Changing the LFP OCP curve

Since changing transport and kinetic parameters did not work, the next step was to directly modify the LFP OCP function.

This is an **empirical calibration**, rather than an attempt to create a more physically accurate electrochemical model.

The Prada2013 parameter set uses an LFP OCP function from Afshar2017:

```python
def LFP_ocp_Afshar2017(sto):
    c1 = -150 * sto
    c2 = -30 * (1 - sto)
    k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)
    return k
```

Inspection of the function showed that `sto` increases towards 1 as the cell discharges. Around `sto=1`, the function gives a voltage of about 2.49 V, which means that it controls the voltage region we are interested in.

The important part is the `-30` rate constant:

```python
-0.9 * np.exp(-30 * (1 - sto))
```

This term creates the steep drop at the end of the OCP curve.

The experiment therefore tested smaller values for this rate constant. A smaller absolute value makes the voltage drop happen over a wider stoichiometry range, which makes the tail less steep.

The script `diagnose_lfp_ocp_rate_constant.py` tested different values across the same C-rate and temperature range as before.

### Results

This parameter behaved very differently from diffusivity.

It could be changed smoothly over a wide range without causing simulation failures or collapse.

| Rate constant  | Successful | In real target (-2.9 to -0.3) | Mean magnitude |
| -------------- | ---------: | ----------------------------: | -------------: |
| -30 (baseline) |        9/9 |                           0/9 |          -3.37 |
| -10            |        9/9 |                           0/9 |          -3.25 |
| -7             |        9/9 |                           0/9 |          -2.99 |
| -5             |        9/9 |                           9/9 |          -2.57 |
| -4             |        9/9 |                           9/9 |          -2.07 |
| **-3**         |    **9/9** |                       **9/9** |      **-1.31** |
| -2             |        9/9 |                           9/9 |          -0.82 |
| -1             |        9/9 |                           9/9 |          -0.39 |

The value **-3** gave the closest overall match.

The real LFP mean is -1.28, while the simulation with rate=-3 gives -1.31. This is a very close match.

The softer values -2 and -1 were not chosen because they went further than necessary. In addition, the -1 setting produced three new outliers, which is a warning sign that the parameter should not be pushed further.

There were also nine outliers at every tested rate, including the original value of -30. These were outside the target zone and were unchanged by this modification. They therefore appear to be an existing issue rather than something introduced by the OCP change.

### Per-bin check

The mean magnitude gives a very good match at rate=-3, but the individual voltage bins are not equally well matched.

For example, at 0.15C and 25°C:

* Deepest bin: synthetic -2.97 vs. real -2.87
* 2.9 to 3.0 V: synthetic -0.55 vs. real -0.31

The deepest bin therefore matches very well, while the shallower bins are still around 1.7 to 2 times too large.

So the OCP change clearly improves the overall magnitude, but it does not completely fix every individual bin.

## Phase 3: Combining the LFP and NMC fixes

The final step was to run the complete simulation pipeline with both fixes:

**NMC**

* Diffusivity divided by 10
* Chen2020 + OKane2022
* Mohtat2020 excluded

**LFP**

* OCP tail rate constant changed from -30 to -3

Other settings from Experiment 20 were kept:

* SOH = 0.8
* C-rate = 0.1 to 0.2C
* 1.5 V cutoff
* Dense `t_interp` sampling

All 498 simulations were successful.

### Results

| Model         | Raw accuracy | Balanced accuracy | LFP recall | NMC recall |
| ------------- | -----------: | ----------------: | ---------: | ---------: |
| Random Forest |       86.27% |        **90.55%** |      0.990 |      0.821 |
| **XGBoost**   |       96.31% |        **97.21%** |      0.990 |      0.955 |

This is a large improvement compared with Experiment 20, where only the NMC mismatch had been fixed.

For Random Forest:

**82.92% → 90.55% balanced accuracy**

For XGBoost:

**52.60% → 97.21% balanced accuracy**

The XGBoost result is especially important. In Experiment 20 it was essentially at chance level. After fixing the LFP magnitude mismatch as well, it reached 97.21%.

This suggests that XGBoost was previously learning from a synthetic distribution that was unrealistic for both chemistries. Fixing only NMC was therefore not enough.

### Checking for class imbalance

The improvement is not simply caused by one class being classified correctly while the other fails.

Both classes have high recall:

* LFP: 0.990
* NMC: 0.821 for Random Forest
* NMC: 0.955 for XGBoost

This gives high balanced accuracy for both models.

The main feature, `dV_dQ_V_3.0_2.9`, also has around 76% feature importance and has almost 100% synthetic coverage for both chemistries. Therefore, the result does not appear to depend on a feature with very limited data coverage.

### Coverage and remaining mismatch

Coverage is still very good:

* LFP: about 99.6%
* NMC: 100% in most bins, down to 83.1% in the deepest bin

The magnitude match is still not perfect.

For example, in the 2.9 to 2.8 V bin:

* Real LFP: -0.71
* Synthetic LFP: -1.55

So the synthetic values are still somewhat too large in several bins.

NMC also still has an uneven magnitude match, as described in Experiment 20.

However, classification accuracy is now high despite these remaining differences. This suggests that the remaining magnitude mismatch is no longer the main problem for classification.

## Overall conclusion

Experiment 21 tested whether the diffusivity approach used successfully for NMC could also fix LFP's magnitude mismatch.

It could not.

For LFP, changing diffusivity and other transport or kinetic parameters had essentially no effect. Instead, changing the rate constant in the fitted LFP OCP curve gave a clear improvement.

The value **-3** gave the closest overall match between synthetic and real LFP data.

Combining this LFP fix with the NMC diffusivity/10 fix from Experiment 20 produced the strongest sim-to-real classification result so far:

* **Random Forest: 90.55% balanced accuracy**
* **XGBoost: 97.21% balanced accuracy**

This is a large improvement from the approximately 50% balanced accuracy seen in the previous experiments.

Both changes should be viewed as **empirical calibrations**, not as proof that these parameter values are physically more accurate.

For NMC, the diffusivity was reduced beyond the typical literature range. For LFP, the fitted OCP curve was directly modified. The purpose was to make the synthetic data better match the real data, rather than to claim a more accurate first-principles electrochemical model.

### Remaining open questions

The following issues were not addressed in this experiment:

* Mohtat2020 is still excluded from the NMC simulations.
* The magnitude match is still uneven between individual bins.
* The fixes have only been tested at SOH=0.8.
* The new settings should be tested at other SOH values before concluding that they generalize.

Full run outputs:

`sim_and_real_raw.csv`

`features/ml_features.csv`

The feature file can be regenerated using `simulate_batteries_lfp_ocp_tuned.py` followed by `evaluate_lfp_ocp_tuned_sim_to_real.py`.
