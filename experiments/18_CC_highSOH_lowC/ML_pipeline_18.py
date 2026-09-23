import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Dynamic directory resolution matching simulate_batteries_18.py and feature_engineering_18.py
DATA_DIR = os.environ.get("DATA_DIR", "data")
RUN_LABEL = os.environ.get("RUN_LABEL", "continuous_discharge_18")
RUN_DIR = os.path.join(DATA_DIR, RUN_LABEL)

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(RUN_DIR, "models"))


def run_ml_pipeline(
    synthetic_csv: str,
    real_csv: str = None,
    pretrained_model_path: str = None,
    save_artifacts: bool = True,
    models_dir: str = MODELS_DIR,
    plot_filename: str = "ml_eval_plots_18.png",
):
    """
    synthetic_csv: feature CSV to train on (uses 100% of data when real_csv is provided).
    real_csv: real experimental feature dataset used strictly as the holdout test set.
    pretrained_model_path: optional model to warm-start/fine-tune.
    save_artifacts: persist the trained model + preprocessing to models_dir.
    models_dir: target directory for artifacts and evaluation plots.
    plot_filename: filename for saving confusion matrix and feature importances.
    """
    if not os.path.exists(synthetic_csv):
        raise FileNotFoundError(f"Synthetic feature dataset not found at: {synthetic_csv}")

    df = pd.read_csv(synthetic_csv)
    df["Chemistry"] = df["Chemistry"].astype(str).str.strip().str.upper()

    metadata_cols = [
        "Battery_ID",
        "Chemistry",
        "Size_Multiplier",
        "SOH",
        "Initial_SOC",
        "Target_Capacity_Ah",
        "Variation_ID",
        "N_Steps",
        "C_Rate",
        "Current [A]",
        "Ambient_Temperature_C",
        "Resistance_Factor",
        "Base_Parameter_Set",
        "V_min [V]",
    ]

    os.makedirs(models_dir, exist_ok=True)
    feature_names_path = os.path.join(models_dir, "feature_names_18.json")
    label_encoder_path = os.path.join(models_dir, "label_encoder_18.joblib")
    feature_imputer_path = os.path.join(models_dir, "feature_imputer_18.joblib")
    feature_scaler_path = os.path.join(models_dir, "feature_scaler_18.joblib")

    if pretrained_model_path and os.path.exists(feature_names_path):
        with open(feature_names_path, "r") as f:
            feature_cols = json.load(f)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
    else:
        feature_cols = [col for col in df.columns if col not in metadata_cols]

    print(f"Identified {len(feature_cols)} candidate synthetic feature columns.")

    if not feature_cols:
        print("No feature columns available -- skipping training.")
        le = LabelEncoder()
        classes = list(le.fit(df["Chemistry"]).classes_)
        return {
            "best_model": None,
            "best_model_name": "(no usable features)",
            "best_accuracy": None,
            "model_accuracies": {},
            "feature_importances": {},
            "classes": classes,
        }

    X = df[feature_cols].copy()
    y = df["Chemistry"].copy()

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    print(f"Target classes mapped: {dict(zip(classes, range(len(classes))))}")

    # -----------------------------------------------------------------------
    # TRAIN / TEST SPLIT & FEATURE ALIGNMENT
    # -----------------------------------------------------------------------
    if real_csv is not None and os.path.exists(real_csv):
        print(f"\n--> [Step 2] Loading real experimental test set from: {real_csv}")
        df_real = pd.read_csv(real_csv)
        df_real["Chemistry"] = df_real["Chemistry"].astype(str).str.strip().str.upper()

        # Restrict features to bins present in BOTH synthetic and real data
        # (Real battery cut off at 2.5V, so sub-2.5V synthetic bins are dropped)
        common_features = [col for col in feature_cols if col in df_real.columns]

        if common_features:
            feature_cols = common_features
            X = X[feature_cols].copy()
            print(f"Aligning feature space: {len(feature_cols)} mutually present voltage bins selected.")
        else:
            print("Warning: No identical bin overlap found between real and synthetic features. Reindexing with NaNs.")

        X_train = X
        y_train = y_encoded

        # Safely reindex real features to match exact order and avoid KeyError
        X_test = df_real.reindex(columns=feature_cols).copy()
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        y_test = le.transform(df_real["Chemistry"])

        print(
            f"Sim-to-Real split: {len(X_train)} synthetic train (100%) | "
            f"{len(X_test)} real test samples (100%)."
        )
    else:
        if real_csv is not None:
            print(f"Warning: Real feature file '{real_csv}' not found. Falling back to Stratified Group Split.")

        print("\n--> Performing 5-fold Stratified Group Split on synthetic data...")
        groups = df["Battery_ID"].values
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(X, y_encoded, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        print(f"Split completed: {len(X_train)} train samples | {len(X_test)} test samples.")

    # -----------------------------------------------------------------------
    # Preprocessing (fitted strictly on training set)
    # -----------------------------------------------------------------------
    lower_bound = X_train.quantile(0.01)
    upper_bound = X_train.quantile(0.99)
    X_train_clipped = X_train.clip(lower=lower_bound, upper=upper_bound, axis=1)
    X_test_clipped = X_test.clip(lower=lower_bound, upper=upper_bound, axis=1)

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_clipped)
    X_test_imputed = imputer.transform(X_test_clipped)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # -----------------------------------------------------------------------
    # Model Benchmarking
    # -----------------------------------------------------------------------
    print("\n" + "=" * 45)
    print("MODEL PERFORMANCE COMPARISON (SIM-TO-REAL)")
    print("=" * 45)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    best_model_name = ""
    best_accuracy = -1.0
    best_model = None
    best_preds = None
    model_accuracies = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
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

    print(f"Detailed Classification Report ({best_model_name}):")
    print(classification_report(y_test, best_preds, target_names=classes, zero_division=0))

    if save_artifacts:
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(best_model, os.path.join(models_dir, "best_battery_classifier_18.joblib"))
        joblib.dump(scaler, feature_scaler_path)
        joblib.dump(imputer, feature_imputer_path)
        joblib.dump(le, label_encoder_path)
        with open(feature_names_path, "w") as f:
            json.dump(feature_cols, f)
        print("Pipeline artifacts saved: model, scaler, imputer, label encoder, and feature names.")

    # -----------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0],
    )
    title_suffix = "Real Holdout Test Set" if real_csv else "Synthetic Test Fold"
    axes[0].set_title(f"Confusion Matrix ({title_suffix}): {best_model_name}")
    axes[0].set_xlabel("Predicted Chemistry")
    axes[0].set_ylabel("True Chemistry")

    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importances = dict(zip(feature_cols, importances))
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_cols[i] for i in indices]
        top_weights = importances[indices]

        sns.barplot(
            x=top_weights,
            y=top_features,
            hue=top_features,
            legend=False,
            ax=axes[1],
            palette="viridis",
        )
        axes[1].set_title(f"Top 10 Feature Importances ({best_model_name})")
        axes[1].set_xlabel("Relative Importance Weight")

    plt.tight_layout()

    if save_artifacts:
        eval_plot_path = os.path.join(models_dir, plot_filename)
        plt.savefig(eval_plot_path, dpi=150)
        print(f"Evaluation plots saved to '{eval_plot_path}'")
    plt.close(fig)

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_accuracy": best_accuracy,
        "model_accuracies": model_accuracies,
        "feature_importances": feature_importances,
        "classes": classes,
    }


if __name__ == "__main__":
    default_synthetic_features = os.path.join(
        RUN_DIR,
        "features",
        "ml_features_continuous_18.csv",
    )
    synthetic_input = os.environ.get("INPUT_FEATURES_CSV", default_synthetic_features)

    default_real_features = os.path.join(
        RUN_DIR,
        "features",
        "ml_features_real_18.csv",
    )
    real_input = os.environ.get("REAL_FEATURES_CSV", default_real_features)

    run_ml_pipeline(
        synthetic_csv=synthetic_input,
        real_csv=real_input,
    )