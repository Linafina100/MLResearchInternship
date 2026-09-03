import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1. Load the LARGE dataset
df = pd.read_csv("stena_training_data.csv")

# 2. Function to calculate dV/dQ (Feature Engineering)
def calculate_dvdq(data):
    data = data.sort_values("Time [s]")
    data['Voltage_Smooth'] = savgol_filter(data['Voltage [V]'], window_length=11, polyorder=3)
    dV = np.diff(data['Voltage_Smooth'])
    dQ = np.diff(data['Capacity [A.h]'])
    dQ[dQ == 0] = 1e-9
    data['dVdQ'] = np.insert(np.abs(dV / dQ), 0, 0)
    return data

# 3. Function to extract exactly 15 features
def extract_15_features(data, num_steps=15):
    capacity_min = data['Capacity [A.h]'].min()
    capacity_max = data['Capacity [A.h]'].max()
    target_capacities = np.linspace(capacity_min, capacity_max, num_steps)
    return np.interp(target_capacities, data['Capacity [A.h]'], data['dVdQ'])

# 4. Process EACH battery individually
X = []
y = []

print("Processing feature extraction for 100 batteries...")
# Group by the unique Battery_ID we created in the generation script
for battery_id, battery_data in df.groupby('Battery_ID'):
    processed_data = calculate_dvdq(battery_data.copy())
    features = extract_15_features(processed_data)
    
    X.append(features)
    # The chemistry is the same for all rows of a specific battery, so we just take the first one
    y.append(battery_data['Chemistry'].iloc[0])

X = np.array(X)
y = np.array(y)

# 5. Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the Random Forest model
print("Training the AI model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 7. Evaluate the model on the unseen test data
y_pred = rf_model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8. Plot Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title("Confusion Matrix: LFP vs NMC")
plt.show()