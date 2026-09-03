import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. Load the previously generated synthetic data
df = pd.read_csv("synthetic_battery_data.csv")

def calculate_dvdq(data):
    # Sort data chronologically to ensure correct diff calculations
    data = data.sort_values("Time [s]")
    
    # 2. Apply Savitzky-Golay filter to smooth the voltage curve
    # window_length=11 and polyorder=3 are robust defaults for battery data
    data['Voltage_Smooth'] = savgol_filter(data['Voltage [V]'], window_length=11, polyorder=3)
    
    # 3. Calculate dV (change in voltage) and dQ (change in capacity)
    dV = np.diff(data['Voltage_Smooth'])
    dQ = np.diff(data['Capacity [A.h]'])
    
    # Prevent division by zero if two data points share the exact same capacity
    dQ[dQ == 0] = 1e-9
    
    # Calculate dV/dQ (absolute value is used for a cleaner positive peak visualization)
    dVdQ = np.abs(dV / dQ)
    
    # np.diff shortens the array by 1, so we pad the beginning with a zero
    data['dVdQ'] = np.insert(dVdQ, 0, 0)
    
    return data

# 4. Clean and calculate the derivative for each chemistry
lfp_data = calculate_dvdq(df[df['Chemistry'] == 'LFP'].copy())
nmc_data = calculate_dvdq(df[df['Chemistry'] == 'NMC'].copy())

# 5. Plot the dV/dQ curve to visually confirm the difference in chemistries
plt.figure(figsize=(10, 5))
plt.plot(lfp_data['Capacity [A.h]'], lfp_data['dVdQ'], label='LFP', color='red', linewidth=2)
plt.plot(nmc_data['Capacity [A.h]'], nmc_data['dVdQ'], label='NMC', color='blue', linestyle='--', linewidth=2)

plt.title("Differential Voltage Analysis (dV/dQ)")
plt.xlabel("Discharged Capacity [Ah]")
plt.ylabel("dV/dQ [V/Ah]")
plt.ylim(0, 1.5) # Limit y-axis to cut off the massive initial spike at the start of discharge
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()