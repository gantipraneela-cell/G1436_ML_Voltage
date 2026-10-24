import pandas as pd
import numpy as np

print("=" * 70)
print("EXPERIMENT 2 - FEATURE CHECK")
print("=" * 70)

X = pd.read_csv("data/preprocessed/mixed_dnn/E2_X_train_raw.csv")

print("\nShape:")
print(X.shape)

print("\nData types:")
print(X.dtypes.value_counts())

# Check non-numeric columns
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNon-numeric columns:")
print(non_numeric)

# Check missing values
missing = X.isnull().sum().sum()

print("\nTotal missing values:")
print(missing)

# Check infinite values
numeric_X = X.select_dtypes(include=[np.number])

infinite_count = np.isinf(numeric_X.values).sum()

print("\nTotal infinite values:")
print(infinite_count)

print("\nFirst 20 columns:")
for i, column in enumerate(X.columns[:20], start=1):
    print(f"{i:3d}. {column}")

print("\nLast 20 columns:")
start = max(0, len(X.columns) - 20)

for i, column in enumerate(X.columns[start:], start=start + 1):
    print(f"{i:3d}. {column}")

print("\n" + "=" * 70)
print("FEATURE CHECK COMPLETED")
print("=" * 70)