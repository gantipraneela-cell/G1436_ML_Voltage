import pandas as pd
from sklearn.model_selection import train_test_split

# Load feature-engineered dataset
df = pd.read_csv("battery_Li_features.csv")

# Separate input features and target
X = df.drop(columns=["average_voltage"])
y = df["average_voltage"]

# 90% training, 10% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

# Save files
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Original dataset:", df.shape)
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)