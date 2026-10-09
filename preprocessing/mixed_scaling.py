import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 60)
print("MIXED DATASET FEATURE SCALING")
print("=" * 60)

# Load data
X_train = pd.read_csv("mixed_X_train.csv")
X_test = pd.read_csv("mixed_X_test.csv")

y_train = pd.read_csv("mixed_y_train.csv")
y_test = pd.read_csv("mixed_y_test.csv")

print("\nBefore scaling:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Scaler
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrame
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

print("\nAfter scaling:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)

# Save
X_train_scaled.to_csv(
    "mixed_X_train_scaled.csv",
    index=False
)

X_test_scaled.to_csv(
    "mixed_X_test_scaled.csv",
    index=False
)

y_train.to_csv(
    "mixed_y_train_scaled.csv",
    index=False
)

y_test.to_csv(
    "mixed_y_test_scaled.csv",
    index=False
)

# Save scaler
joblib.dump(
    scaler,
    "mixed_scaler.pkl"
)

print("\nFiles saved:")
print("mixed_X_train_scaled.csv")
print("mixed_X_test_scaled.csv")
print("mixed_y_train_scaled.csv")
print("mixed_y_test_scaled.csv")
print("mixed_scaler.pkl")

print("\n" + "=" * 60)
print("MIXED SCALING COMPLETED")
print("=" * 60)