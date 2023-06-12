#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:47:45 2023

@author: polykko

This code imports lifetime data, performs normalization and single exponential fit
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns

# Import data from CSV file using pd.read_csv

data = pd.read_csv('5hpf_200_3.csv', skiprows=0)

# Extract x and y values from the DataFrame
x = data['X0'][0:51]
y = data['Y0'][0:51]

# Normalize y
y_normalized = (y - y.min()) / (y.max() - y.min())
# Add gate delay in ps
gate_delay = 200

# Define the exponential function with an offset
def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Perform the exponential fit
initial_guess = [1, 0.1, 1]  # Adjust the initial guess if necessary
params, _ = curve_fit(exponential_func, x, y_normalized, maxfev=1000)

# Extract the values of a, b, and c
a, b, c = params

# Generate the fitted curve using the obtained parameters
x_fit = np.linspace(min(x), max(x), 100)
y_fit = exponential_func(x_fit, *params)

# Calculate the lifetime in ns
lifetime = np.round(((1/b) * gate_delay)/1000,1)

# Plotting the data and the fit
sns.set_theme()
plt.rcParams["figure.dpi"] = 500
plt.scatter(x*gate_delay/1000, y_normalized, s=15, c='cornflowerblue', label='Intensity Measurement')
plt.plot(x_fit*gate_delay/1000, y_fit, 'r-', linewidth=1.5, alpha=0.8, label='Exponential Fit')
plt.plot([], [], ' ', label=f"Lifetime = {round(1/b * gate_delay/1000, 1)} ns")
plt.plot([], [], ' ', label=f"Gate width = {gate_delay} ps")
plt.xlabel('Time (ns)')
plt.ylabel('Normalized Intensity (a.u.)')
#plt.title(f"Lifetime Measurements")
plt.legend()
#plt.text(0.05, 0.95, 'Lifetime = 3.6 ns', transform=plt.gca().transAxes, ha='left', va='top')

plt.show()