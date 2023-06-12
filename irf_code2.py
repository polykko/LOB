#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:51:24 2023

@author: polykko

This code imports IRF measurement data, normalizes it, and performs measurements of FWHM
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
sns.set_theme()

# Get the current working directory
cwd = os.getcwd()

# Get the path to the CSV file
filename = 'irf9.csv'
filepath = os.path.join(cwd, filename)

# Set the gate delay in ps
gate_delay = 50

# Load data from CSV file, skipping the first row
df = pd.read_csv(filepath, skiprows=1)

# Extract x and y values from the data frame
x = df['x'].to_numpy()
y = df['y'].to_numpy()
y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

# Create an interpolation function
interp_fn = interp1d(x, y_norm, kind='linear')

# Generate additional points
# CAREFUL, ARRAYS ARE OVERWRITTEN
x = np.linspace(x.min(), x.max(), num=20000)
y = interp_fn(x)

# Find maximum y value and its index
max_y = np.max(y)
max_y_index = np.argmax(y)

# Calculate half-maximum value
hm_value = max_y / 2

# Find indexes where y is closest to half-maximum value
left_index = np.argmin(np.abs(y[:max_y_index] - hm_value)) 
right_index = np.argmin(np.abs(y[max_y_index:] - hm_value)) + max_y_index

# Calculate width at half-maximum
whm = x[right_index] - x[left_index]
whm_ps = round(whm * gate_delay, 3)
print("Width at half-maximum:", whm_ps, "ps")

# Plot the data
fig, ax = plt.subplots()
plt.rcParams["figure.dpi"] = 500
ax.plot(x, y, 'cornflowerblue')
ax.plot([x[left_index], x[right_index]], [y[left_index], y[right_index]], 'ro-', markersize=3, linewidth=1)
ax.set_xlabel('Frame Number')
ax.set_ylabel('Normalized Intensity (a.u.)')
#plt.grid()
ax.set_title('IRF for 1000 ps Gate Width')

# Show the plot and print the width at half-maximum
plt.show()
