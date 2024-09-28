# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load an example dataset (Iris)
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy and display the classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample Data for Regression
data = {
    'age': [25, 35, 45, 50, 23, 34, 28, 46],
    'salary': [50000, 60000, 70000, 80000, 45000, 54000, 62000, 75000]
}

# Create features and target variables
X = np.array(data['age']).reshape(-1, 1)  # Features
y = np.array(data['salary'])  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor
reg = DecisionTreeRegressor(random_state=42)

# Fit the model on the training data
reg.fit(X_train, y_train)

# Predict on the test data
y_pred = reg.predict(X_test)

# Calculate and display Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Predict new values
new_age = np.array([[30]])
predicted_salary = reg.predict(new_age)
print(f"Predicted salary for age 30: {predicted_salary[0]}")


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Fit a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
