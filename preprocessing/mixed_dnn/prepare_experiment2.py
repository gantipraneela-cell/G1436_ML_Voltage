import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/preprocessed/mixed_dnn")
SOURCE_FILE = Path("data/raw/battery_mixed_data.csv")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DNN EXPERIMENT 2 - PREPROCESSING AND DATA SPLIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load original dataset
# ------------------------------------------------------------
df = pd.read_csv(SOURCE_FILE)
print("\nOriginal dataset:")
print("Shape:", df.shape)

# ------------------------------------------------------------
# 2. Remove rows with missing target
# ------------------------------------------------------------

df = df.dropna(subset=["average_voltage"]).reset_index(drop=True)

print("\nAfter removing missing target values:")
print("Shape:", df.shape)

# ------------------------------------------------------------
# 3. One-hot encode working_ion
# ------------------------------------------------------------

if "working_ion" in df.columns:

    df = pd.get_dummies(
        df,
        columns=["working_ion"],
        dtype=int
    )

    print("\nworking_ion one-hot encoded.")

# ------------------------------------------------------------
# 4. Separate features and target
# ------------------------------------------------------------

X = df.drop(columns=["average_voltage"])
y = df["average_voltage"]

print("\nFeature matrix:")
print("X shape:", X.shape)

print("\nTarget:")
print("y shape:", y.shape)

# ------------------------------------------------------------
# 5. Create regression bins for stratified splitting
# ------------------------------------------------------------
#
# We use quantile bins so that the general target distribution
# is represented in train, validation, and test.
#
# duplicates='drop' prevents errors if quantile boundaries
# are identical.
# ------------------------------------------------------------

y_bins = pd.qcut(
    y,
    q=10,
    labels=False,
    duplicates="drop"
)

print("\nTarget stratification bins created:")
print("Number of bins:", y_bins.nunique())

# ------------------------------------------------------------
# 6. First split:
#    90% temporary training pool
#    10% final test set
# ------------------------------------------------------------

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y_bins
)

print("\nAfter train/test split:")
print("Training pool:", X_temp.shape)
print("Test:", X_test.shape)

# ------------------------------------------------------------
# 7. Create bins again for the training pool
# ------------------------------------------------------------

y_temp_bins = pd.qcut(
    y_temp,
    q=10,
    labels=False,
    duplicates="drop"
)

# ------------------------------------------------------------
# 8. Second split:
#    90% training
#    10% validation
#
#    Overall:
#    81% train
#     9% validation
#    10% test
# ------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.10,
    random_state=42,
    stratify=y_temp_bins
)

print("\nFinal split:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

# ------------------------------------------------------------
# 9. Save processed full dataset
# ------------------------------------------------------------

processed = pd.concat(
    [
        X.reset_index(drop=True),
        y.reset_index(drop=True)
    ],
    axis=1
)

processed.to_csv(
    DATA_DIR / "battery_mixed_experiment2_processed.csv",
    index=False
)
# ------------------------------------------------------------
# 10. Save train/validation/test data
# ------------------------------------------------------------

X_train.to_csv(DATA_DIR / "E2_X_train_raw.csv", index=False)
X_val.to_csv(DATA_DIR / "E2_X_val_raw.csv", index=False)
X_test.to_csv(DATA_DIR / "E2_X_test_raw.csv", index=False)

y_train.to_csv(DATA_DIR / "E2_y_train.csv", index=False)
y_val.to_csv(DATA_DIR / "E2_y_val.csv", index=False)
y_test.to_csv(DATA_DIR / "E2_y_test.csv", index=False)

# ------------------------------------------------------------
# 11. Print target distributions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

for name, target in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test)
]:

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    print("Samples :", len(target))
    print("Minimum :", f"{target.min():.6f}")
    print("Maximum :", f"{target.max():.6f}")
    print("Mean    :", f"{target.mean():.6f}")
    print("Median  :", f"{target.median():.6f}")
    print("99%     :", f"{target.quantile(0.99):.6f}")

    print("\nCounts above thresholds:")

    for threshold in [5, 6, 8, 10, 15, 20, 30]:
        print(
            f"> {threshold:2d}:",
            int((target > threshold).sum())
        )

# ------------------------------------------------------------
# 12. Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 2 PREPROCESSING COMPLETED")
print("=" * 70)

print("\nFiles created:")
print("battery_mixed_experiment2_processed.csv")
print("E2_X_train_raw.csv")
print("E2_X_val_raw.csv")
print("E2_X_test_raw.csv")
print("E2_y_train.csv")
print("E2_y_val.csv")
print("E2_y_test.csv")