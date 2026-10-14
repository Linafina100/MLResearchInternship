# Hyperparameter Tuning for SOC 0.1–0.4 After the Artifact Fix

## What this experiment tests

Experiment 14 removed the termination step artifact by excluding the final raw transition from each battery. For SOC 0.1–0.4, this reduced the accuracy from the artifact affected 94.74% to 73.68% for Random Forest and 75.79% for XGBoost. Only one feature bin remained: `dV_dQ_V_3.2_3.1`.

This experiment tests whether the remaining accuracy can be improved by tuning the model hyperparameters instead of using the current manually selected settings.

The experiment only focuses on SOC 0.1–0.4. The other SOC intervals are not considered here because they are either already close to maximum accuracy or outside the scope of this experiment. The preprocessing steps were kept unchanged.

The same raw data from experiment 03 was used, together with `feature_engineering.create_features_by_voltage_bins(exclude_final_transition=True)`, to reproduce the dataset from experiment 14.

### Method

The preprocessing steps were included in an `sklearn` pipeline so that they were fitted separately within each cross validation fold. These steps included replacing infinite values with NaN, clipping values at the 1st and 99th percentiles, median imputation, and standard scaling.

`GridSearchCV` was then used to test different hyperparameter combinations using 5 fold `StratifiedGroupKFold`, with `Battery_ID` used for grouping. The hyperparameters were selected using only the training data. The best configuration for each model was then evaluated once on the held out test set.

As a check, the baseline models were also evaluated using the same split. This reproduced the results from experiment 14.

## Results

| Model         | Baseline test | Tuned CV | Tuned test |
| ------------- | ------------: | -------: | ---------: |
| Random Forest |        73.68% |   77.45% | **80.00%** |
| XGBoost       |        75.79% |   77.19% | **80.00%** |

**Best hyperparameters:**

* **Random Forest:** `max_depth=5`, `min_samples_leaf=10`, `n_estimators=300`
* **XGBoost:** `learning_rate=0.01`, `max_depth=3`, `n_estimators=50`, `subsample=1.0`

The test set contained 95 samples. The baseline Random Forest correctly classified 70 samples, while both tuned models correctly classified 76.

## Interpretation

Hyperparameter tuning produced a modest improvement. Random Forest increased from 73.68% to 80.00%, while XGBoost increased from 75.79% to 80.00%.

The tuned models also performed slightly better on the held out test set than in the inner cross validation, which suggests that the tuning did not simply produce a configuration that performed well during cross validation but failed on unseen data. However, the test set is relatively small, so the improvement should still be interpreted with some caution.

Both models performed best with more conservative settings than the original configurations. The Random Forest used shallower trees and a larger minimum number of samples per leaf, while XGBoost used a lower learning rate and fewer trees. This suggests that the original settings may have been too complex for a dataset with only 377 training samples and a single feature.

Overall, hyperparameter tuning improves the results, but it does not recover the accuracy seen before the artifact was removed. The 94.74% accuracy was influenced by the termination step artifact and should therefore not be treated as genuine model performance. The current 80% accuracy is a more realistic result for this feature set.

Further improvements are therefore more likely to come from adding useful information, such as additional voltage ranges or features, rather than further tuning the same models on this single feature.
