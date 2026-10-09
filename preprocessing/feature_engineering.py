import pandas as pd

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

INPUT_FILE = "battery_Li_cleaned.csv"
OUTPUT_FILE = "battery_Li_features.csv"

TARGET = "average_voltage"

df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:")
print(df.shape)

# Separate target
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nTarget:")
print(TARGET)

print("\nNumber of input features:")
print(X.shape[1])

# Missing values
print("\nTotal missing feature values:")
print(X.isnull().sum().sum())

# Constant features
constant_features = X.columns[X.nunique() <= 1]

print("\nConstant features:")
print(len(constant_features))

# Duplicate feature pairs
duplicate_pairs = 0

columns = X.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        if X[columns[i]].equals(X[columns[j]]):
            duplicate_pairs += 1

print("\nDuplicate feature pairs:")
print(duplicate_pairs)

# Very low variance features
numeric_X = X.select_dtypes(include="number")

variance = numeric_X.var()

very_low_variance = variance[variance < 1e-6]

print("\nVery-low-variance features:")
print(very_low_variance)

# Combine features and target
df_final = pd.concat([X, y], axis=1)

# Save
df_final.to_csv(OUTPUT_FILE, index=False)

print("\nFinal shape:")
print(df_final.shape)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)