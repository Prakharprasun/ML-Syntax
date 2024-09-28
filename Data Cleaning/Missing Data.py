import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Sample DataFrame with missing values
data = {'age': [25, np.nan, 30, 29, np.nan],
        'salary': [50000, 54000, np.nan, 62000, 58000],
        'gender': ['M', 'F', np.nan, 'F', 'M']}

df = pd.DataFrame(data)

# Example: Mean imputation for numerical data
imputer_mean = SimpleImputer(strategy='mean')
df['age'] = imputer_mean.fit_transform(df[['age']])

# Example: Median imputation for numerical data
imputer_median = SimpleImputer(strategy='median')
df['salary'] = imputer_median.fit_transform(df[['salary']])

# Example: Most frequent (mode) imputation for categorical data
imputer_mode = SimpleImputer(strategy='most_frequent')
df['gender'] = imputer_mode.fit_transform(df[['gender']])

# Example: Constant imputation for categorical data
imputer_constant = SimpleImputer(strategy='constant', fill_value='Unknown')
df['gender'] = imputer_constant.fit_transform(df[['gender']])