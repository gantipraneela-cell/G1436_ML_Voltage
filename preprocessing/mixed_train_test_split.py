import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("MIXED DATASET TRAIN / TEST SPLIT")
print("=" * 60)

# Load final mixed dataset
df = pd.read_csv("battery_mixed_final.csv")

TARGET = "average_voltage"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nOriginal dataset:")
print(df.shape)

print("\nInput features:")
print(X.shape[1])

# 90% train / 10% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# Save
X_train.to_csv("mixed_X_train.csv", index=False)
X_test.to_csv("mixed_X_test.csv", index=False)

y_train.to_csv("mixed_y_train.csv", index=False)
y_test.to_csv("mixed_y_test.csv", index=False)

print("\nFiles saved:")
print("mixed_X_train.csv")
print("mixed_X_test.csv")
print("mixed_y_train.csv")
print("mixed_y_test.csv")

print("\n" + "=" * 60)
print("MIXED TRAIN / TEST SPLIT COMPLETED")
print("=" * 60)