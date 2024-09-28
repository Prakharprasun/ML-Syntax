# dictionary to map class to number
class_to_num = {"Class" : {'Setosa': 1, 'Versicolor': 2, 'Virginica': 3}}

# use replace function to map class to number
df.replace(class_to_num, inplace=True)
print(df["Class"].value_counts()) # print the number of data points to check if each class has been changed to a number

from sklearn.preprocessing import LabelEncoder

# Sample data
data = {'color': ['red', 'blue', 'green', 'blue', 'green']}
df = pd.DataFrame(data)

# Initialize the encoder
label_encoder = LabelEncoder()

# Fit and transform the data
df['color_encoded'] = label_encoder.fit_transform(df['color'])

print(df)

import pandas as pd

# Sample data
data = {'color': ['red', 'blue', 'green', 'blue', 'green']}
df = pd.DataFrame(data)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['color'])

print(df_encoded)

from sklearn.preprocessing import OneHotEncoder

# Sample data
data = {'color': ['red', 'blue', 'green', 'blue', 'green']}
df = pd.DataFrame(data)

# Initialize the encoder
onehot_encoder = OneHotEncoder(sparse=False)

# Fit and transform the data
encoded_data = onehot_encoder.fit_transform(df[['color']])

# Convert to DataFrame
df_encoded = pd.DataFrame(encoded_data, columns=onehot_encoder.categories_[0])

print(df_encoded)

from sklearn.preprocessing import OrdinalEncoder

# Sample data
data = {'size': ['small', 'medium', 'large', 'medium', 'small']}
df = pd.DataFrame(data)

# Initialize the encoder
ordinal_encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])

# Fit and transform the data
df['size_encoded'] = ordinal_encoder.fit_transform(df[['size']])

print(df)

# Sample data
data = {'color': ['red', 'blue', 'green', 'blue', 'green']}
df = pd.DataFrame(data)

# Frequency encoding
freq_encoding = df['color'].value_counts(normalize=False)
df['color_encoded'] = df['color'].map(freq_encoding)

print(df)

import category_encoders as ce

# Sample data with target variable
data = {'color': ['red', 'blue', 'green', 'blue', 'green'],
        'target': [1, 0, 1, 0, 1]}
df = pd.DataFrame(data)

# Initialize the target encoder
target_encoder = ce.TargetEncoder(cols=['color'])

# Fit and transform the data based on the target variable
df['color_encoded'] = target_encoder.fit_transform(df['color'], df['target'])

print(df)
