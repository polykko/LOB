#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:09:09 2023

@author: polykko

The code imports 3 .csv files, normalizes and plots the curves
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Define the file paths
file_paths = ['irf8.csv', 'irf9.csv', 'irf10.csv']

# Initialize lists to store x and y data for each file
x_data = []
y_data = []

# Read data from each file
for file_path in file_paths:
    # Read the file using pandas, skipping the header
    df = pd.read_csv(file_path, skiprows=1)
    
    # Extract x and y columns
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    
    # Append the x and y data to the overall lists
    x_data.append(x)
    y_data.append(y)

# Normalize the y data for each file
normalized_y_data = []
for y in y_data:
    normalized_y = (y - np.min(y)) / (np.max(y) - np.min(y))
    normalized_y_data.append(normalized_y)

# Plot the curves
sns.set_theme()
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['cornflowerblue', 'coral', 'forestgreen'])
plt.rcParams["figure.dpi"] = 500
# Alpha controls line opacity [0.0, 1.0]
for x, y in zip(x_data, normalized_y_data):
    plt.plot(x, y, linewidth=1, alpha=0.8)


# Add labels and a legend
plt.xlabel('Frame Number')
plt.ylabel('Normalized Intensity (a.u.)')
plt.title('Normalized IRF Curves for 1000 ps Gate Width')
#plt.legend(['irf8', 'irf9', 'irf10'])

# Display the plot
#plt.grid()
plt.show()