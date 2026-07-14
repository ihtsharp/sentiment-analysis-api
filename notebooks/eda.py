import pandas as pd

# Load dataset
df = pd.read_csv("data/data.csv")

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset shape
print("\nShape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Dataset information
print("\nInfo:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Rating distribution
print("\nRatings:")
print(df["overall"].value_counts())