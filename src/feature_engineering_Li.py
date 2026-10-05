import pandas as pd
import numpy as np

INPUT_FILE = "battery_Li_cleaned.csv"
OUTPUT_FILE = "battery_Li_features.csv"

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:")
print(df.shape)

# --------------------------------------------------
# 2. Separate target
# --------------------------------------------------
target = "average_voltage"

X = df.drop(columns=[target])
y = df[target]

print("\nTarget:")
print(target)

print("\nNumber of input features:")
print(X.shape[1])

# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------
missing = X.isnull().sum().sum()

print("\nTotal missing feature values:")
print(missing)

# --------------------------------------------------
# 4. Check constant features
# --------------------------------------------------
constant_features = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

print("\nConstant features:")
print(len(constant_features))

if constant_features:
    print(constant_features)

# --------------------------------------------------
# 5. Check duplicate feature columns
# --------------------------------------------------
duplicate_features = []

columns = X.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        if X[columns[i]].equals(X[columns[j]]):
            duplicate_features.append(
                (columns[i], columns[j])
            )

print("\nDuplicate feature pairs:")
print(len(duplicate_features))

if duplicate_features:
    for pair in duplicate_features:
        print(pair)

# --------------------------------------------------
# 6. Feature variance
# --------------------------------------------------
variance = X.var()

low_variance = variance[variance < 1e-6]

print("\nVery-low-variance features:")
print(len(low_variance))

if len(low_variance) > 0:
    print(low_variance)

# --------------------------------------------------
# 7. Feature correlation with target
# --------------------------------------------------
correlation = X.corrwith(y).sort_values(
    key=lambda x: abs(x),
    ascending=False
)

print("\nTop 15 features correlated with average_voltage:")
print(correlation.head(15))

# --------------------------------------------------
# 8. Save feature-engineered dataset
# --------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved feature-engineered dataset:")
print(OUTPUT_FILE)

print("\nFinal shape:")
print(df.shape)

print("\nFEATURE ENGINEERING CHECK COMPLETE")
print("=" * 60)