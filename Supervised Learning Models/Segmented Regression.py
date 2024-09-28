# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Create a sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 1.5, 2, 3, 4, 7, 9, 10, 15, 20])

# Define breakpoints for the piecewise segments
breakpoint = 5

# Create piecewise variables
X_segmented = np.column_stack((X, (X > breakpoint) * (X - breakpoint)))

# Fit the segmented regression model
model = sm.OLS(y, sm.add_constant(X_segmented)).fit()

# Generate predictions
y_pred = model.predict(sm.add_constant(X_segmented))

# Plot the results
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Segmented Regression Fit')
plt.title('Segmented Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
