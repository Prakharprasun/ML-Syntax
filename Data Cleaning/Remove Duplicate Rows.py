import pandas as pd

# Create a sample DataFrame
data = {
    'A': [1, 2, 2, 3, 4],
    'B': [5, 6, 6, 7, 8],
    'C': [9, 10, 10, 11, 12]
}
df = pd.DataFrame(data)

# Remove duplicate rows
df_unique = df.drop_duplicates()

# Display the result
print("DataFrame after removing duplicate rows:")
print(df_unique)
