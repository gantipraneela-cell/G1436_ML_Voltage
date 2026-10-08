import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv("../data/features/battery_Li_feature_engineered.csv")

print("Original dataset shape:", df.shape)

# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["average_voltage"])
y = df["average_voltage"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# CREATE VOLTAGE RANGES FOR STRATIFIED SPLITTING
# ============================================================

voltage_bins = [-np.inf, 2, 3, 4, 5, np.inf]

voltage_labels = [
    "Below_2V",
    "2_to_3V",
    "3_to_4V",
    "4_to_5V",
    "Above_5V"
]

voltage_groups = pd.cut(
    y,
    bins=voltage_bins,
    labels=voltage_labels
)

print("\nVoltage range distribution:")
print(voltage_groups.value_counts().sort_index())

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=voltage_groups
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n====================================")
print("TRAIN / TEST SPLIT RESULTS")
print("====================================")

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

# ============================================================
# CHECK VOLTAGE DISTRIBUTION
# ============================================================

train_groups = pd.cut(
    y_train,
    bins=voltage_bins,
    labels=voltage_labels
)

test_groups = pd.cut(
    y_test,
    bins=voltage_bins,
    labels=voltage_labels
)

print("\nTraining voltage distribution:")
print(train_groups.value_counts().sort_index())

print("\nTesting voltage distribution:")
print(test_groups.value_counts().sort_index())

# ============================================================
# SAVE FILES
# ============================================================

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nFiles saved successfully.")