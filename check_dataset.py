import pandas as pd

file_path = "datasets/Crop_recommendation.csv"

df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())