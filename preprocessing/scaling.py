import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 60)
print("FEATURE SCALING")
print("=" * 60)

# Load train/test data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

print("\nBefore scaling:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Create scaler
scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using training scaler
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
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

print("\nTarget:")
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# Save
X_train_scaled.to_csv("X_train_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv", index=False)

y_train.to_csv("y_train_scaled.csv", index=False)
y_test.to_csv("y_test_scaled.csv", index=False)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("\nFiles saved:")
print("X_train_scaled.csv")
print("X_test_scaled.csv")
print("y_train_scaled.csv")
print("y_test_scaled.csv")
print("scaler.pkl")

print("\n" + "=" * 60)
print("SCALING COMPLETED")
print("=" * 60)