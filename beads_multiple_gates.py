#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:58:07 2023

@author: polykko

This code imports beads lifetime data, normalizes it and performs signgle exponential fit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
sns.set_theme()

# Define the exponential function with an offset
def exponential_with_offset(x, a, b, c):
    return a * np.exp(-b * x) + c

# List of CSV file names
csv_files = ["beads200.csv", "beads500.csv", "beads1000.csv"]

# Initialize lists to store the normalized data
x_data = []
y_data_normalized = []
gate=[0.2,0.5,1]
labels=["200","500","1000"]

# Find the common x-axis range
x_min = float("inf")
x_max = float("-inf")

# Import and normalize the data from each CSV file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the X and Y columns
    x = df["X0"]
    y = df["Y0"]
    
    # Update the common x-axis range
    x_min = min(x_min, x.min())
    x_max = max(x_max, x.max())
    
    # Normalize the Y data
    y_normalized = (y - y.min()) / (y.max() - y.min())
    
    # Append the normalized data to the lists
    x_data.append(x)
    y_data_normalized.append(y_normalized)

# Fit exponential function to each set of data and plot
fig, ax = plt.subplots()

for i in range(len(csv_files)):
    # Filter out NaN and inf values from the data
    valid_indices = np.isfinite(x_data[i]) & np.isfinite(y_data_normalized[i])
    x_valid = x_data[i][valid_indices] * gate[i]
    y_valid_normalized = y_data_normalized[i][valid_indices]

    # Fit exponential with offset to the valid data
    popt, pcov = curve_fit(exponential_with_offset, x_valid, y_valid_normalized)
    
    # Extract the values of a, b, and c
    a, b, c = popt
    
    # Generate a curve using the fitted parameters
    curve_x = np.linspace(x_min, x_max, 100)
    curve_y = exponential_with_offset(curve_x, *popt)
    
    # Plot the original data and the fitted curve
    ax.scatter(x_valid, y_valid_normalized, s=10, alpha=0.7, label=f"Gate width {round(gate[i]*1000)} ps, Lifetime {round(1/b, 1)} ns")
    ax.plot(curve_x,curve_y, linestyle="--", alpha=0.9, label="Exponential fit")
    print(b)


    #f"Number: {numbers[2]}"
# Set labels and title for the plot
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Normalized Intensity (a.u.)")
ax.set_xlim(0,10)

#ax.set_title("Exponential Fit of Normalized Data")

# Add a legend
#ax.legend(title="Gate Width (ps)")
ax.legend()

# Display the plot
plt.rcParams["figure.dpi"] = 500
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['cornflowerblue', 'coral', 'forestgreen'])
plt.show()
