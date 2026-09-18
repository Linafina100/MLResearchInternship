# Real LFP vs. NMC classification test: sim to real

## Motivation

So far, all results in this project have been based on PyBaMM simulations. The goal, however, is to eventually use the classifier on real battery measurements. This means an important test is whether a model trained on simulated data can classify real batteries.

The original version of this experiment instead trained and tested the model using only real data. This was not a sim to real test, so the experiment was redesigned to correctly evaluate how well simulated data transfers to real data.

## Data

* **LFP:** `DownloadedData/ENSAYOSBATERIA/` contains data from one real LFP cell (LiFePO4 32700, 6000 mAh) with 579 discharge cycles. Each cycle consists of rest, a 6 A discharge to 2.5 V, and rest, with approximately 1 Hz sampling.
* **NMC:** `DownloadedData/Dataset_2_NCM_battery/` contains data from three real NMC cells tested at different temperatures (`CY25-05_1-#1`, `CY35-05_1-#1`, `CY45-05_1-#1`), with 1,782 cycles in total. The data comes from Zhu et al., *Nature Communications* (2022).
* **Synthetic data:** The four simulated datasets from experiment 03 were combined. They cover the SOC intervals 0.7–1.0, 0.5–0.8, 0.3–0.6, and 0.1–0.4. Combining them gives a wider voltage range, which is needed to cover the full discharge curves in the real data.

## Method

The synthetic and real data were first combined into one dataset and marked as either synthetic or real using a `DataKind` column. Features were then extracted once using `feature_engineering_continuous.py`.

This was done so that both datasets use exactly the same feature columns. If the features were extracted separately, the synthetic and real datasets could end up with different voltage bins.

After feature extraction, the data was separated again:

* **Training:** 996 synthetic LFP + 943 synthetic NMC batteries
* **Testing:** 577 real LFP + 1,782 real NMC cycles

The models were trained on **100% synthetic data** and tested on **100% real data**. No real data was used for training.

A total of 10 voltage bins survived the feature selection, covering `3.4–3.3 V` to `2.5–2.4 V`.

## Results

| Model         | Accuracy |
| ------------- | -------: |
| Random Forest |   24.84% |
| XGBoost       |   24.84% |

The real test set contains 577 LFP and 1,782 NMC cycles, corresponding to 24.5% LFP and 75.5% NMC.

Both models therefore performed almost exactly like a model that always predicts LFP. The classification report confirms this:

* **LFP recall:** 100%
* **NMC recall:** 1%

In other words, almost all real NMC batteries were classified as LFP.

## Interpretation

The results show that the classifier trained on PyBaMM simulations **does not transfer well to the real battery data** with the current pipeline.

The model performs well on simulated test data, but this performance does not carry over to real cells. The large difference suggests that the dV/dQ patterns learned from the simulations are different from those found in the real NMC data.

This result is also consistent with experiment 16, which tested whether a model trained on one real data source could generalize to another real data source. That experiment also showed very poor generalization.

Together, these experiments show that the current dV/dQ voltage bin features have not been shown to generalize beyond the type of data they were trained on.

This is important when interpreting the high accuracy reported in experiments 03, 06, 11, 12, 14, and 15. Those results were obtained using simulated data and therefore show how well the model works on data generated in the same simulation environment. They do **not** show that the model can classify real batteries.

## Not yet done

The next step is to investigate why the simulated and real dV/dQ patterns differ, especially for NMC. Possible reasons include:

* The PyBaMM NMC parameters may not represent the real cells well.
* The real NMC data has much coarser and more variable sampling than the simulated data.
* The real batteries were tested using different C rates and discharge protocols than the simulations.
* The simulated data may not contain enough variation in battery parameters.

A possible next step is therefore to increase the variation in the simulations, for example by using more NMC parameter sets, a wider range of C rates, and a wider range of SOH values. This can show whether more diverse simulated data improves the transfer to real batteries.

