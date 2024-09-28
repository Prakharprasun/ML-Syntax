from sklearn.feature_selection import VarianceThreshold

# Create a sample DataFrame
data = {
    'A': [1, 2, 2, 3, 4],
    'B': [5, 5, 5, 5, 5],
    'C': [9, 10, 10, 11, 12],
    'D': [1, 2, 1, 2, 1]  # Low variance column
}
df = pd.DataFrame(data)

# Remove columns with low variance (threshold = 0.1)
threshold = 0.1
selector = VarianceThreshold(threshold)
df_filtered = df.loc[:, selector.fit(df).get_support()]

# Display the result
print("\nDataFrame after removing low variance columns:")
print(df_filtered)
