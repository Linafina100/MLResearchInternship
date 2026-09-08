import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_ml_pipeline(
    synthetic_csv: str,
    real_csv: str = None,
    save_artifacts: bool = True
):
    """
    Parameters:
    synthetic_csv : Path to the feature-engineered dataset generated from PyBaMM simulations.
    real_csv (optional): Path to experimental factory test data. If provided, models will train on
    synthetic data and test exclusively on real data (Sim-to-Real).
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
    """
    metadata_cols = ['Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    print(f"Identified {len(feature_cols)} feature columns for training.")

    X = df[feature_cols].copy()
    y = df['Chemistry'].copy()

    """
    2. INFINITIES & ARTIFACTS
    a) During resting intervals, transition phases or measurement delays, 
    current may stop or sensor reading migh not register a chnage in capacity => Q=0
    Division by zero (dQ -> 0) creates +/- inf, 
    when scaler etc is applied it makes the entire column mean into inf => value error and the model crash.
    => So we convert them directly to NaNs.

    b)In feature engineering we use zero padding to will up the shorter runs so all have 15 variables. 
    This means that in dV/dQ, 0.0 often indicates missing pulses (early cutoff) rather than a
    true physical flat curve. Which can cause problems for the classification. 
    Convert 0.0 back to NaN so the imputer/XGBoost handles them as missing observations rather 
    than false plateau features. 
    This wont effect real plateau features because they are never exactly 0.0.
    """
    X.replace([np.inf, -np.inf], np.nan, inplace=True) #convert +/- into NaN
    X.replace(0.0, np.nan, inplace=True) #convert 0.0 to NaN

    """
    3. LABEL ENCODING
    Convert text labels (LFP, NMC) to numeric (0, 1) for model training.
    """
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
        X_test.replace(0.0, np.nan, inplace=True)
        y_test = le.transform(df_real['Chemistry'])
        print(f"Sim-to-Real split: {len(X_train)} synthetic train samples | {len(X_test)} real test samples.")
    else:
        # If no real data is provided yet, use Stratified Grouped Split on simulation data.
        # Grouping by Battery_ID prevents augmented/sliced battery cycles from leaking into test.
        print("\n--> [Step 2] Performing 80/20 Stratified Train-Test Split on synthetic data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
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
    """
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
    6. MODEL DEFINITION
    We will train and compare two tree-based ensemble models: Random Forest and XGBoost.
    Hyperparameters are set to reasonable defaults, but can be tuned further for optimal performance.
    """
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

    """
    7. TRAINING & EVALUATION
    If evalutating on synthetic data, this tests generalization across different cell sizes, SOH, or intital SOC.
    If evaluating on real data, this tests sim-to-real generalization.
    """
    print("\n" + "=" * 45)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 45)

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
    # Standard training mode:
    run_ml_pipeline(synthetic_csv="ml_features_data.csv")
    
    # Future real-world validation mode (uncomment when factory test data is ready):
    # run_ml_pipeline(
    #     synthetic_csv="ml_features_data.csv",
    #     real_csv="real_experimental_data.csv"
    # )