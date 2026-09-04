import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_ml_pipeline(input_csv):
    print(f"Loading feature data from {input_csv}...\n")
    df = pd.read_csv(input_csv)
    
    # Identify all columns containing the dV/dQ steps
    feature_cols = [col for col in df.columns if col.startswith('dV_dQ_step_')]
    
    X = df[feature_cols]
    y = df['Chemistry']
    
    # Convert text labels (LFP, NMC) to numeric (0, 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 80/20 Train-Test Split (as described in the article)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Standardization (Scaling)
    # Extremely important for SVM, as it relies on distances between data points
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the models we want to compare
    models = {
    "Support Vector Classifier": SVC(kernel='rbf', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}
    
    # Train and evaluate each model
    print("-" * 40)
    print("MODEL EVALUATION (Accuracy)")
    print("-" * 40)
    
    best_model_name = ""
    best_accuracy = 0
    best_model = None
    accuracies = {}
    
    for name, model in models.items():
        # Train the model on the scaled training data
        model.fit(X_train_scaled, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        print(f"{name:<28}: {acc*100:.2f}%")
        
        # Save the best performing model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print("-" * 40)
    print(f"BEST MODEL: {best_model_name} with {best_accuracy*100:.2f}%\n")
    
    # Show a detailed report for the winning model
    print(f"Detailed report for {best_model_name}:")
    y_pred_best = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))
    
    # Confusion Matrix for the best model
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.ylabel('Actual Chemistry')
    plt.xlabel('Predicted Chemistry')
    plt.show()

if __name__ == "__main__":
    # Run the pipeline
    run_ml_pipeline("ml_features_data.csv")