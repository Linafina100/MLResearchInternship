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

# Directory for trained ML artifacts (models, scalers, encoders, feature lists).
MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def run_ml_pipeline(
    synthetic_csv: str,
    real_csv: str = None,
    pretrained_model_path: str = None,
    save_artifacts: bool = True
):
    """
    synthetic_csv: feature CSV to train/evaluate on.
    real_csv: optional real experimental test set (sim-to-real mode).
    pretrained_model_path: optional Phase 1 model to warm-start/fine-tune
        (loads its matching scaler/imputer/encoder/feature list too).
    save_artifacts: persist the trained model + preprocessing to MODELS_DIR.
    """
    df = pd.read_csv(synthetic_csv)

    # Features = everything not in metadata_cols. 
    # Target capacity and metadata/text columns (like Base_Parameter_Set) 
    # are excluded to prevent leakage or breaking numeric ML models.
    # Fine-tuning uses feature_names.json to align columns, padding missing bins with NaN.

    ### ADDED VARIATION ID
    metadata_cols = [
        'Battery_ID', 'Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC',
        'Target_Capacity_Ah', 'N_Steps',
        'Ambient_Temperature_C', 'Resistance_Factor', 'Base_Parameter_Set', 'Variation_ID',
    ]
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.json")
    label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    feature_imputer_path = os.path.join(MODELS_DIR, "feature_imputer.joblib")
    feature_scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")

    if pretrained_model_path and os.path.exists(feature_names_path):
        with open(feature_names_path, "r") as f:
            feature_cols = json.load(f)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
    else:
        feature_cols = [col for col in df.columns if col not in metadata_cols]
    print(f"Identified {len(feature_cols)} feature columns for training.")

    if not feature_cols:
        # Returns an empty result dict so the pipeline can safely move on.
        print("No feature columns available -- skipping training (no usable signal at this configuration).")
        le = LabelEncoder()
        classes = list(le.fit(df['Chemistry']).classes_)
        return {
            "best_model": None,
            "best_model_name": "(no usable features)",
            "best_accuracy": None,
            "model_accuracies": {},
            "feature_importances": {},
            "classes": classes,
        }

    X = df[feature_cols].copy()
    y = df['Chemistry'].copy()

    # dQ->0 during rests creates +-inf in dV/dQ; convert to NaN. A real
    # dV/dQ of exactly 0.0 (e.g. LFP's plateau) is genuine signal and must
    # NOT be treated as missing.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Encode LFP/NMC to 0/1. Fine-tuning reuses the Phase 1 encoder so
    # classes can't silently swap.
    if pretrained_model_path and os.path.exists(label_encoder_path):
        le = joblib.load(label_encoder_path)
        y_encoded = le.transform(y)
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    print(f"Target classes mapped: {dict(zip(classes, range(len(classes))))}")

    # real_csv given -> train on sim, test on real (sim-to-real). Otherwise
    # an 80/20 stratified GROUP split (grouped by Battery_ID so no
    # battery's samples leak across train/test).
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
        print("\n--> [Step 2] Performing ~80/20 Stratified Group Split on synthetic data...")
        groups = df['Battery_ID'].values
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(X, y_encoded, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        print(f"Split completed: {len(X_train)} train samples | {len(X_test)} test samples.")

    # Fit only on X_train to avoid leakage: clip outliers to the 1st/99th
    # percentile (pulse start/end can be unstable), median-impute,
    # standardize. Fine-tuning reuses Phase 1's imputer/scaler instead of
    # refitting, to keep the base model's trained scale intact.
    if (
        pretrained_model_path
        and os.path.exists(feature_imputer_path)
        and os.path.exists(feature_scaler_path)
    ):
        imputer = joblib.load(feature_imputer_path)
        scaler = joblib.load(feature_scaler_path)
        X_train_imputed = imputer.transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        X_train_scaled = scaler.transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
    else:
        lower_bound = X_train.quantile(0.01)
        upper_bound = X_train.quantile(0.99)
        X_train_clipped = X_train.clip(lower=lower_bound, upper=upper_bound, axis=1)
        X_test_clipped = X_test.clip(lower=lower_bound, upper=upper_bound, axis=1)

        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_test_imputed = imputer.transform(X_test_clipped)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

    # Standard mode: train+compare RF and XGBoost from scratch. Fine-tuning:
    # warm-start 50 more trees onto a saved Phase 1 model -- RF's
    # n_estimators is a NEW TOTAL (increment it), XGBoost's is boosting
    # rounds ON TOP of the existing booster (set directly to 50, don't
    # increment).
    print("\n" + "=" * 45)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 45)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\n--> Loading pre-trained Phase 1 model from: {pretrained_model_path}")
        best_model = joblib.load(pretrained_model_path)
        n_new_estimators = 50

        if isinstance(best_model, XGBClassifier):
            # xgb_model= continuation verifies that training features match the original model. 
            # X_train_scaled is a plain ndarray (imputer/scaler strip column names), 
            # which avoids strict XGBoost name-matching errors and relies purely on positional alignment.
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
        model_accuracies = {best_model_name: best_accuracy}
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
        model_accuracies = {}

        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Generate predictions on unseen test set
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            model_accuracies[name] = acc

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

    # Save model + preprocessing + feature list for deployment.
    if save_artifacts:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_model, os.path.join(MODELS_DIR, "best_battery_classifier.joblib"))
        joblib.dump(scaler, feature_scaler_path)
        joblib.dump(imputer, feature_imputer_path)
        joblib.dump(le, label_encoder_path)
        with open(feature_names_path, "w") as f:
            json.dump(feature_cols, f)
        print("Pipeline artifacts saved: model, scaler, imputer, label encoder, and feature names.")

    # Confusion matrix + feature-importance plot.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix: {best_model_name}')
    axes[0].set_xlabel('Predicted Chemistry')
    axes[0].set_ylabel('True Chemistry')

    # Feature Importances (Top 10 most influential steps)
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importances = dict(zip(feature_cols, importances))
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_cols[i] for i in indices]
        top_weights = importances[indices]

        sns.barplot(x=top_weights, y=top_features, ax=axes[1], palette="viridis")
        axes[1].set_title(f'Top 10 Feature Importances ({best_model_name})')
        axes[1].set_xlabel('Relative Importance Weight')

    plt.tight_layout()
    plt.show()

    # Dict (not just best_model) so callers can pull per-model
    # accuracy/importances without scraping stdout.
    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_accuracy": best_accuracy,
        "model_accuracies": model_accuracies,
        "feature_importances": feature_importances,
        "classes": classes,
    }


# New 
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    # Path to the features generated by feature_engineering_calce.py
    CALCE_FEATURES_CSV = os.path.join(
        REPO_ROOT, "data", "default", "features", "ml_features_25_steps.csv"
    )

    # Run the ML pipeline
    run_ml_pipeline(synthetic_csv=CALCE_FEATURES_CSV)

# OLD
# if __name__ == "__main__":
    # Standard training mode (Phase 2 SOC-bin baseline, trained from scratch):
   # run_ml_pipeline(synthetic_csv=os.path.join("data", "default", "features", "ml_features_25_steps.csv")) #update for every step count and SOC

    # Fine-tuning mode (uncomment once the above baseline has been saved and
    # truncated real-world factory data is ready): loads
    # models/best_battery_classifier.joblib + its feature_names.json/scaler/
    # imputer/label_encoder saved alongside it, and warm-starts with 50 new
    # estimators.
    # run_ml_pipeline(
    #     synthetic_csv="real_experimental_data.csv",
    #     pretrained_model_path=os.path.join(MODELS_DIR, "best_battery_classifier.joblib"),
    # )

    # Future real-world validation mode (uncomment when factory test data is ready):
    # run_ml_pipeline(
    #     synthetic_csv="ml_features_data.csv",
    #     real_csv="real_experimental_data.csv"
    # )