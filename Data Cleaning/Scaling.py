from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Sample Data
data = {'age': [25, 35, 45, 50],
        'salary': [50000, 60000, 70000, 80000]}
df = pd.DataFrame(data)

# Initialize Min-Max Scaler
scaler = MinMaxScaler()

# Fit and transform the data
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_scaled)


from sklearn.preprocessing import StandardScaler
import pandas as pd

# Sample Data
data = {'age': [25, 35, 45, 50],
        'salary': [50000, 60000, 70000, 80000]}
df = pd.DataFrame(data)

# Initialize Standard Scaler
scaler = StandardScaler()

# Fit and transform the data
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_scaled)

from sklearn.preprocessing import Normalizer

# Sample Data
data = {'age': [25, 35, 45, 50],
        'salary': [50000, 60000, 70000, 80000]}
df = pd.DataFrame(data)

# Initialize Normalizer
scaler = Normalizer()

# Fit and transform the data
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_scaled)