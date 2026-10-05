import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("battery_mixed_cleaned.csv")

X = df.drop(columns=["average_voltage"])
y = df["average_voltage"]

# Remove non-numeric columns if present
X = X.select_dtypes(include="number")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42
)

X_train.to_csv("mixed_X_train.csv", index=False)
X_test.to_csv("mixed_X_test.csv", index=False)
y_train.to_csv("mixed_y_train.csv", index=False)
y_test.to_csv("mixed_y_test.csv", index=False)

print("=" * 60)
print("MIXED DATA TRAIN / TEST SPLIT")
print("=" * 60)
print("Original:", df.shape)
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)