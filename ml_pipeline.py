import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier

# Load the synthetic battery data
df = pd.read_csv("synthetic_battery_data.csv")

# Function to calculate dV/dQ (Feature Engineering)
def calculate_dvdq(data):
    data = data.sort_values("Time [s]")
    # Smooth the curve using Savitzky-Golay filter
    data['Voltage_Smooth'] = savgol_filter(data['Voltage [V]'], window_length=11, polyorder=3)
    dV = np.diff(data['Voltage_Smooth'])
    dQ = np.diff(data['Capacity [A.h]'])
    
    # Prevent division by zero
    dQ[dQ == 0] = 1e-9
    
    # Calculate dV/dQ and pad the array to match original length
    data['dVdQ'] = np.insert(np.abs(dV / dQ), 0, 0)
    return data

# Process the data to get the dV/dQ values
lfp_data = calculate_dvdq(df[df['Chemistry'] == 'LFP'].copy())
nmc_data = calculate_dvdq(df[df['Chemistry'] == 'NMC'].copy())

# Function to extract exactly 15 features (steps) from a curve
def extract_15_features(data, num_steps=15):
    capacity_min = data['Capacity [A.h]'].min()
    capacity_max = data['Capacity [A.h]'].max()
    
    # Create 15 evenly spaced measurement points
    target_capacities = np.linspace(capacity_min, capacity_max, num_steps)
    
    # Interpolate to find the exact dV/dQ values at these 15 points
    return np.interp(target_capacities, data['Capacity [A.h]'], data['dVdQ'])

# Create our training dataset (X = features, y = labels)
X = []
y = []

# Add LFP features
X.append(extract_15_features(lfp_data))
y.append("LFP")

# Add NMC features
X.append(extract_15_features(nmc_data))
y.append("NMC")

# Convert to numpy arrays for the ML model
X = np.array(X)
y = np.array(y)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

print("Model training complete!")
print("Model's prediction for the first battery (should be LFP):", rf_model.predict([X[0]]))
print("Model's prediction for the second battery (should be NMC):", rf_model.predict([X[1]]))