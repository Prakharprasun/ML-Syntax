# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Create or load a dataset
# Example: Creating a simple dataset
data = {'feature1': [1, 2, 3, 4, 100],
        'feature2': [2, 3, 4, 5, -100]}

df = pd.DataFrame(data)

# Instantiate the Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)

# Fit the model to the data
iso_forest.fit(df)

# Predict anomalies (-1 indicates anomaly, 1 indicates normal point)
df['anomaly'] = iso_forest.predict(df)

# Print the results
print(df)
