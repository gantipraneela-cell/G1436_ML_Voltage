import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("mixed_X_train.csv")
X_test = pd.read_csv("mixed_X_test.csv")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

X_train_scaled.to_csv("mixed_X_train_scaled.csv", index=False)
X_test_scaled.to_csv("mixed_X_test_scaled.csv", index=False)

print("=" * 60)
print("MIXED DATA SCALING")
print("=" * 60)

print("Before scaling:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\nAfter scaling:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)

print("\nScaling completed successfully!")