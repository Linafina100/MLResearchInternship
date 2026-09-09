import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_ml_pipeline(
    synthetic_csv: str,
    real_csv: str = None,
    pretrained_model_path: str = None,
    save_artifacts: bool = True
):
    """
    Parameters:
    synthetic_csv : Path to the feature-engineered dataset generated from PyBaMM simulations.
    real_csv (optional): Path to experimental factory test data. If provided, models will train on
    synthetic data and test exclusively on real data (Sim-to-Real).
    pretrained_model_path (optional): Path to a Phase 1 .joblib model to warm-start/fine-tune
    instead of training from scratch. When set, the matching feature_names.json,
    feature_scaler.joblib, feature_imputer.joblib, and label_encoder.joblib (saved alongside
    it by a prior save_artifacts=True run) are loaded too, so Phase 2 data goes through the
    exact same feature ordering and preprocessing the base model was trained under.
    save_artifacts :If True, dumps the best model, scaler, and feature list for recycling plant deployment.
    """
    df = pd.read_csv(synthetic_csv)

    """
    1. DYNAMIC FEATURE SELECTION
    Instead of hardcoding 'col.startswith("dV_dQ_step_")', we drop known metadata.
    This ensures any new features added in feature engineering (e.g., rest voltage,
    internal resistance, temperature) are automatically included.
    We remove metadata columns that are not features: Battery_ID, Chemistry, Size_Multiplier,
    SOH, Initial_SOC so that the model cant "cheat" by using labels.
    Target_Capacity_Ah and N_Steps are excluded too: the paper deliberately varies cell
    capacity across all chemistries so that chemistry identification cannot be shortcut
    via capacity ("To ensure that the cathode chemistries are not identified by their
    different cell capacities..."). Leaving Target_Capacity_Ah in X would let the model
    do exactly that. N_Steps is constant within any single per-step-count CSV, so it
    carries no information anyway, but it's metadata, not a physical feature.

    When fine-tuning, feature ordering is instead loaded from the Phase 1 run's
    feature_names.json so Phase 2 data lines up with what the base model was
    trained on -- any Phase 1 feature absent from this Phase 2 dataset (e.g. a
    SOC bin the truncated real-world data never reaches) is added as an all-NaN
    column so the imputer can still fill it, rather than shifting every other
    column's position.
    """
    metadata_cols = [
        'Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC',
        'Target_Capacity_Ah', 'N_Steps',
    ]
    if pretrained_model_path and os.path.exists("feature_names.json"):
        with open("feature_names.json", "r") as f:
            feature_cols = json.load(f)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
    else:
        feature_cols = [col for col in df.columns if col not in metadata_cols]
    print(f"Identified {len(feature_cols)} feature columns for training.")

    X = df[feature_cols].copy()
    y = df['Chemistry'].copy()

    """
    2. INFINITIES & MISSING STEPS
    During resting intervals, transition phases or measurement delays, current may stop
    or the sensor reading might not register a change in capacity => Q=0. Division by
    zero (dQ -> 0) creates +/- inf; when the scaler etc. is applied this makes the
    entire column mean become inf => value error and the model crashes.
    => So we convert them directly to NaNs.

    Note: feature_engineering_advanced.py no longer zero-pads missing entries. A
    battery that never reaches a given SOC bin (or hits an early voltage cutoff)
    simply has fewer dV_dQ_SOC_* keys, and pandas naturally fills the missing
    columns with NaN when the rows are combined into a DataFrame. So NaN already
    correctly means "missing/unreached" here, and a real dV/dQ value of exactly
    0.0 (e.g. from the LFP plateau) is genuine physical signal, not a missing-data
    artifact — it must NOT be overwritten with NaN.
    """
    X.replace([np.inf, -np.inf], np.nan, inplace=True) #convert +/- into NaN

    """
    3. LABEL ENCODING
    Convert text labels (LFP, NMC) to numeric (0, 1) for model training.
    When fine-tuning, reuse the Phase 1 encoder instead of fitting a new one --
    a freshly-fit encoder could assign LFP/NMC to different integers than the
    base model was trained against, silently swapping the classes.
    """
    if pretrained_model_path and os.path.exists("label_encoder.joblib"):
        le = joblib.load("label_encoder.joblib")
        y_encoded = le.transform(y)
    else:
        le = LabelEncoder() #so that we can later convert back to text labels for confusion matrix and classification report
        y_encoded = le.fit_transform(y)
    classes = list(le.classes_) #preserves mapping internally as an array for later reference
    print(f"Target classes mapped: {dict(zip(classes, range(len(classes))))}")

    """
    4. SPLIT SEPARATION (SIMULATION VS. REAL EXPERIMENTAL DATA)
    If real experimental data is provided, we will train on synthetic data and test exclusively on real data.
    If no real data is provided, we will perform a stratified train-test split on the synthetic data to evaluate model performance.
    """

    if real_csv is not None:
        print(f"\n--> [Step 2] Loading real experimental test set from: {real_csv}")
        df_real = pd.read_csv(real_csv)
        
        # Enforce identical feature alignment between synthetic and real sets
        X_train = X
        y_train = y_encoded
        X_test = df_real[feature_cols].copy()
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        y_test = le.transform(df_real['Chemistry'])
        print(f"Sim-to-Real split: {len(X_train)} synthetic train samples | {len(X_test)} real test samples.")
    else:
        # If no real data is provided yet, use a Stratified Grouped Split on simulation data.
        # Grouping by Battery_ID prevents augmented/sliced battery cycles from leaking into test
        # (each battery's samples land entirely in either train or test, never split across both).
        print("\n--> [Step 2] Performing ~80/20 Stratified Group Split on synthetic data...")
        groups = df['Battery_ID'].values
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(X, y_encoded, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        print(f"Split completed: {len(X_train)} train samples | {len(X_test)} test samples.")

    """
    5. PREPROCESSING
    We fit strictly on X_train to avoid data leakage, then transform X_test.
    a) Outlier clipping: Clip extreme values to the 1st and 99th percentiles to
    reduce the influence of outliers. At the very beginning and end of the pulse,
    the current may not have stabilized yet, which can create extreme dV/dQ values
    that are not representative of the chemistry.
    b) Median imputation: Fill NaN values with the median of each feature column.
    c) Feature standardization: Scale features to have zero mean and unit variance.
    Not that important for the current models, but good practice for future models (e.g., SVM, Neural Networks).

    When fine-tuning, reuse the Phase 1 imputer/scaler instead of fitting new
    ones on Phase 2 data -- the base model's trees split on the specific scale
    it was trained under, so refitting here would shift that scale out from
    under it rather than genuinely warm-starting it.
    """
    if (
        pretrained_model_path
        and os.path.exists("feature_imputer.joblib")
        and os.path.exists("feature_scaler.joblib")
    ):
        imputer = joblib.load("feature_imputer.joblib")
        scaler = joblib.load("feature_scaler.joblib")
        X_train_imputed = imputer.transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        X_train_scaled = scaler.transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
    else:
        #a) Outlier Clipping
        lower_bound = X_train.quantile(0.01)
        upper_bound = X_train.quantile(0.99)
        X_train_clipped = X_train.clip(lower=lower_bound, upper=upper_bound, axis=1)
        X_test_clipped = X_test.clip(lower=lower_bound, upper=upper_bound, axis=1)

        #b) Median imputer
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_test_imputed = imputer.transform(X_test_clipped)

        #c)Feature standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

    """
    6/7. MODEL TRAINING, OR FINE-TUNING A PHASE 1 MODEL
    Standard mode trains and compares two tree-based ensemble models: Random
    Forest and XGBoost, with hyperparameters set to reasonable defaults.

    Fine-tuning mode loads a Phase 1 model and warm-starts it with 50 new
    estimators trained on Phase 2 data, instead of training from scratch.
    Note the two libraries' "how many more trees" semantics differ:
    - RandomForestClassifier's warm_start=True treats n_estimators as the NEW
      TOTAL to grow toward, so incrementing it by 50 relative to what the
      loaded model already has correctly adds exactly 50 new trees.
    - XGBClassifier's xgb_model=<existing booster> continues boosting for
      n_estimators MORE rounds on top of the passed-in booster -- it is not a
      new total. Incrementing it by 50 the same way as RandomForest would
      train far more than 50 additional trees (150+50=200 extra, not 50), so
      it's set directly to 50 here instead.
    """
    print("\n" + "=" * 45)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 45)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\n--> Loading pre-trained Phase 1 model from: {pretrained_model_path}")
        best_model = joblib.load(pretrained_model_path)
        n_new_estimators = 50

        if isinstance(best_model, XGBClassifier):
            # xgb_model= continuation validates that the new training data's
            # feature names match what the booster was originally fit on.
            # X_train_scaled is a plain ndarray (imputer/scaler strip column
            # names), which matches how this pipeline always fits models --
            # if a baseline is ever fit on a named DataFrame instead, this
            # will raise "training data did not have the following fields".
            booster = best_model.get_booster()
            best_model.n_estimators = n_new_estimators  # additional boosting rounds for THIS fit() call
            best_model.fit(X_train_scaled, y_train, xgb_model=booster)
        elif isinstance(best_model, RandomForestClassifier):
            best_model.warm_start = True
            best_model.n_estimators += n_new_estimators  # warm_start grows toward this new total
            best_model.fit(X_train_scaled, y_train)
        else:
            raise TypeError(
                f"Don't know how to fine-tune a {type(best_model).__name__} -- "
                "only RandomForestClassifier and XGBClassifier are supported."
            )

        best_model_name = f"Fine-Tuned ({type(best_model).__name__})"
        best_preds = best_model.predict(X_test_scaled)
        best_accuracy = accuracy_score(y_test, best_preds)
        print(f"{best_model_name:<25}: Accuracy = {best_accuracy * 100:.2f}%")
    else:
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        }

        best_model_name = ""
        best_accuracy = 0.0
        best_model = None
        best_preds = None

        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Generate predictions on unseen test set
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)

            print(f"{name:<25}: Accuracy = {acc * 100:.2f}%")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_model = model
                best_preds = y_pred

    print("=" * 45)
    print(f"Top Performer: {best_model_name} ({best_accuracy * 100:.2f}% Accuracy)\n")

    # Detailed Classification Metrics
    print(f"Detailed Classification Report ({best_model_name}):")
    print(classification_report(y_test, best_preds, target_names=classes))

    """
    8. PERSISTENCE
    Save the best model, scaler, imputer, label encoder, and feature names for future deployment in a factory setting.
    Kanske inte viktigt att spara label encoder och feature names om man inte ska köra modellen på nya data, 
    men kan vara bra att ha om man vill köra modellen på nya data i framtiden.
    """
    if save_artifacts:
        joblib.dump(best_model, "best_battery_classifier.joblib")
        joblib.dump(scaler, "feature_scaler.joblib")
        joblib.dump(imputer, "feature_imputer.joblib")
        joblib.dump(le, "label_encoder.joblib")
        with open("feature_names.json", "w") as f:
            json.dump(feature_cols, f)
        print("Pipeline artifacts saved: model, scaler, imputer, label encoder, and feature names.")

    """
    9. VISUALIZATION (CONFUSION MATRIX & FEATURE IMPORTANCES)
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix: {best_model_name}')
    axes[0].set_xlabel('Predicted Chemistry')
    axes[0].set_ylabel('True Chemistry')

    # Feature Importances (Top 10 most influential steps)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_cols[i] for i in indices]
        top_weights = importances[indices]

        sns.barplot(x=top_weights, y=top_features, ax=axes[1], palette="viridis")
        axes[1].set_title(f'Top 10 Feature Importances ({best_model_name})')
        axes[1].set_xlabel('Relative Importance Weight')

    plt.tight_layout()
    plt.show()

    return best_model


if __name__ == "__main__":
    # Standard training mode (Phase 2 SOC-bin baseline, trained from scratch):
    run_ml_pipeline(synthetic_csv="ml_features_25_steps.csv") #update for every step count and SOC

    # Fine-tuning mode (uncomment once the above baseline has been saved and
    # truncated real-world factory data is ready): loads
    # best_battery_classifier.joblib + its feature_names.json/scaler/imputer/
    # label_encoder saved alongside it, and warm-starts with 50 new estimators.
    # run_ml_pipeline(
    #     synthetic_csv="real_experimental_data.csv",
    #     pretrained_model_path="best_battery_classifier.joblib",
    # )

    # Future real-world validation mode (uncomment when factory test data is ready):
    # run_ml_pipeline(
    #     synthetic_csv="ml_features_data.csv",
    #     real_csv="real_experimental_data.csv"
    # )