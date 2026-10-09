import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

# Load feature-engineered dataset
df = pd.read_csv("battery_Li_features.csv")

# Target
TARGET = "average_voltage"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nOriginal dataset:", df.shape)

# 90% training, 10% testing
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
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nFiles saved:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT COMPLETED")
print("=" * 60)