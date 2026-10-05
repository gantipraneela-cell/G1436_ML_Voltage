import pandas as pd
import numpy as np

print("=" * 60)
print("FEATURE ENGINEERING - MIXED DATASET")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

input_file = "battery_mixed_cleaned.csv"
output_file = "battery_mixed_feature_engineered.csv"

df = pd.read_csv(input_file)

print("\nOriginal shape:")
print(df.shape)

# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

target = "average_voltage"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found!")

print("\nTarget:")
print(target)

# ------------------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=[target])
y = df[target]

print("\nNumber of input features:")
print(X.shape[1])

# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

missing_total = X.isnull().sum().sum()

print("\nTotal missing feature values:")
print(missing_total)

if missing_total > 0:
    print("\nFeatures containing missing values:")
    print(X.isnull().sum()[X.isnull().sum() > 0])

# ------------------------------------------------------------
# 5. REMOVE CONSTANT FEATURES
# ------------------------------------------------------------

constant_features = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

print("\nConstant features:")
print(len(constant_features))

if constant_features:
    print(constant_features)

    X = X.drop(columns=constant_features)

# ------------------------------------------------------------
# 6. FIND DUPLICATE FEATURES
# ------------------------------------------------------------

print("\nChecking for duplicate feature pairs...")

duplicate_features = set()
columns = X.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        col1 = columns[i]
        col2 = columns[j]

        if X[col1].equals(X[col2]):
            duplicate_features.add(col2)

print("Duplicate feature pairs:")
print(len(duplicate_features))

if duplicate_features:
    print(list(duplicate_features))

    X = X.drop(columns=list(duplicate_features))

# ------------------------------------------------------------
# 7. VERY LOW VARIANCE FEATURES
# ------------------------------------------------------------

numeric_X = X.select_dtypes(include=np.number)

if numeric_X.shape[1] > 0:

    variances = numeric_X.var()

    low_variance = variances[variances < 1e-6]

    print("\nVery-low-variance features:")
    print(len(low_variance))

    if len(low_variance) > 0:
        print(low_variance)

else:
    print("\nNo numeric features found.")

# ------------------------------------------------------------
# 8. HIGH CORRELATION CHECK
# ------------------------------------------------------------

print("\nChecking highly correlated features...")

numeric_X = X.select_dtypes(include=np.number)

if numeric_X.shape[1] > 1:

    corr_matrix = numeric_X.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    highly_correlated = []

    for column in upper.columns:
        correlated = upper[column][upper[column] > 0.95]

        for row in correlated.index:
            highly_correlated.append(
                (row, column, corr_matrix.loc[row, column])
            )

    print("Highly correlated feature pairs (> 0.95):")
    print(len(highly_correlated))

    if highly_correlated:
        for pair in highly_correlated[:20]:
            print(
                f"{pair[0]} <-> {pair[1]} : "
                f"{pair[2]:.4f}"
            )

else:
    print("Not enough numeric features for correlation analysis.")

# ------------------------------------------------------------
# 9. IDENTIFY NON-NUMERIC FEATURES
# ------------------------------------------------------------

non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()

print("\nNon-numeric features:")
print(len(non_numeric))

if non_numeric:
    print(non_numeric)

# ------------------------------------------------------------
# 10. SAVE FEATURE-ENGINEERED DATASET
# ------------------------------------------------------------

X_final = X.copy()

X_final[target] = y

X_final.to_csv(output_file, index=False)

# ------------------------------------------------------------
# 11. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nFinal shape:")
print(X_final.shape)

print("\nFinal number of input features:")
print(X_final.shape[1] - 1)

print("\nTarget:")
print(target)

print("\nOutput file:")
print(output_file)

print("\nFirst 5 rows:")
print(X_final.head())