# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data (you can replace this with your own data)
data = {
    'age': [25, 35, 45, 50, 40],
    'salary': [50000, 60000, 70000, 80000, 75000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Independent (X) and Dependent (y) Variables
X = df[['age']]  # Feature (independent variable)
y = df['salary']  # Target (dependent variable)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression Model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Print model coefficients (slope and intercept)
print(f"Intercept: {model.intercept_}")
print(f"Coefficient (Slope): {model.coef_}")

# Calculate and display performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Predicting new values
new_age = np.array([[30]])  # Replace with your data
predicted_salary = model.predict(new_age)
print(f"Predicted Salary for age 30: {predicted_salary[0]}")
