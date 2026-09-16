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

# Default directory for trained ML artifacts (models, scalers, encoders, feature lists )
MODELS_DIR = os.environ.get("MODELS_DIR", "models")


def run_ml_pipeline(
    synthetic_csv: str,
    real_csv: str = None,
    pretrained_model_path: str = None,
    save_artifacts: bool = True,
    models_dir: str = MODELS_DIR,
    plot_filename: str = "evaluation_metrics_plot.png",
):
    """
    synthetic_csv: feature CSV to train/evaluate on.
    real_csv: optional real experimental test set (sim-to-real mode).
    pretrained_model_path: optional model to warm-start/fine-tune.
    save_artifacts: persist the trained model + preprocessing to models_dir.
    models_dir: target directory for artifacts and evaluation plots.
    plot_filename: filename for saving confusion matrix and feature importances.
    """
    if not os.path.exists(synthetic_csv):
        raise FileNotFoundError(f"Feature dataset not found at: {synthetic_csv}")

    df = pd.read_csv(synthetic_csv)

    # Exclude all identifiers, ground truths, physical cell sizing, and runtime
    # operating parameters so the classifier is forced to learn solely from
    # continuous dV/dQ voltage dynamics rather than shortcuts.
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
    feature_names_path = os.path.join(models_dir, "feature_names.json")
    label_encoder_path = os.path.join(models_dir, "label_encoder.joblib")
    feature_imputer_path = os.path.join(models_dir, "feature_imputer.joblib")
    feature_scaler_path = os.path.join(models_dir, "feature_scaler.joblib")

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
        print(
            "No feature columns available -- skipping training (no usable"
            " signal)."
        )
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

    # Convert mathematical infinities from sensor/division artifacts to NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Encode target classes (LFP -> 0, NMC -> 1)
    if pretrained_model_path and os.path.exists(label_encoder_path):
        le = joblib.load(label_encoder_path)
        y_encoded = le.transform(y)
    else:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    print(f"Target classes mapped: {dict(zip(classes, range(len(classes))))}")

    # Train/Test Split
    if real_csv is not None:
        print(
            f"\n--> [Step 2] Loading real experimental test set from: {real_csv}"
        )
        df_real = pd.read_csv(real_csv)
        X_train = X
        y_train = y_encoded
        X_test = df_real[feature_cols].copy()
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        y_test = le.transform(df_real["Chemistry"])
        print(
            f"Sim-to-Real split: {len(X_train)} synthetic train | {len(X_test)}"
            " real test samples."
        )
    else:
        print(
            "\n--> [Step 2] Performing Stratified Group Split on synthetic"
            " data..."
        )
        groups = df["Battery_ID"].values
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(X, y_encoded, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        print(
            f"Split completed: {len(X_train)} train samples | {len(X_test)}"
            " test samples."
        )

    # Preprocessing: Fit strictly on X_train to prevent leakage
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
        X_train_clipped = X_train.clip(
            lower=lower_bound, upper=upper_bound, axis=1
        )
        X_test_clipped = X_test.clip(
            lower=lower_bound, upper=upper_bound, axis=1
        )

        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train_clipped)
        X_test_imputed = imputer.transform(X_test_clipped)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

    print("\n" + "=" * 45)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 45)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(
            f"\n--> Loading pre-trained model from: {pretrained_model_path}"
        )
        best_model = joblib.load(pretrained_model_path)
        n_new_estimators = 50

        if isinstance(best_model, XGBClassifier):
            booster = best_model.get_booster()
            best_model.n_estimators = n_new_estimators
            best_model.fit(X_train_scaled, y_train, xgb_model=booster)
        elif isinstance(best_model, RandomForestClassifier):
            best_model.warm_start = True
            best_model.n_estimators += n_new_estimators
            best_model.fit(X_train_scaled, y_train)
        else:
            raise TypeError(
                f"Unsupported model type: {type(best_model).__name__}"
            )

        best_model_name = f"Fine-Tuned ({type(best_model).__name__})"
        best_preds = best_model.predict(X_test_scaled)
        best_accuracy = accuracy_score(y_test, best_preds)
        model_accuracies = {best_model_name: best_accuracy}
        print(f"{best_model_name:<25}: Accuracy = {best_accuracy * 100:.2f}%")
    else:
        # colsample_bytree set to 1.0 so XGBoost does not subsample when feature counts are compact
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=1.0,
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
    print(
        f"Top Performer: {best_model_name} ({best_accuracy * 100:.2f}%"
        " Accuracy)\n"
    )

    print(f"Detailed Classification Report ({best_model_name}):")
    print(classification_report(y_test, best_preds, target_names=classes))

    if save_artifacts:
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(
            best_model, os.path.join(models_dir, "best_battery_classifier.joblib")
        )
        joblib.dump(scaler, feature_scaler_path)
        joblib.dump(imputer, feature_imputer_path)
        joblib.dump(le, label_encoder_path)
        with open(feature_names_path, "w") as f:
            json.dump(feature_cols, f)
        print(
            "Pipeline artifacts saved: model, scaler, imputer, label encoder,"
            " and feature names."
        )

    # -----------------------------------------------------------------------
    # PLOTTING: Confusion Matrix & Feature Importances
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Confusion Matrix
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
    axes[0].set_title(f"Confusion Matrix: {best_model_name}")
    axes[0].set_xlabel("Predicted Chemistry")
    axes[0].set_ylabel("True Chemistry")

    # Right: Top Feature Importances
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importances = dict(zip(feature_cols, importances))
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_cols[i] for i in indices]
        top_weights = importances[indices]

        sns.barplot(
            x=top_weights, y=top_features, ax=axes[1], palette="viridis"
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
    # Standard single-run test execution
    input_features_csv = os.path.join(
        "data",
        "continuous_discharge",
        "features",
        "ml_features_continuous.csv",
    )
    run_ml_pipeline(synthetic_csv=input_features_csv)