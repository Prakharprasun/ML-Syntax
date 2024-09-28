# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Create a sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 1.5, 2, 3, 4, 7, 9, 10, 15, 20])

# Fit the Lowess model
lowess = sm.nonparametric.lowess(y, X, frac=0.3)  # frac is the fraction of data used for smoothing

# Extract the smoothed values
X_lowess = lowess[:, 0]
y_lowess = lowess[:, 1]

# Plot the results
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_lowess, y_lowess, color='red', label='Lowess Regression Fit')
plt.title('Locally Weighted Regression (Lowess)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
