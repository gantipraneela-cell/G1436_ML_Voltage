import pandas as pd

print("=" * 60)
print("MIXED DATASET PREPARATION")
print("=" * 60)

INPUT_FILE = "battery_mixed_cleaned.csv"
OUTPUT_FILE = "battery_mixed_final.csv"

TARGET = "average_voltage"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:")
print(df.shape)

# Drop total_atoms
if "total_atoms" in df.columns:
    df = df.drop(columns=["total_atoms"])

# Separate target
y = df[TARGET]

# Remove target from features
X = df.drop(columns=[TARGET])

# One-hot encode working_ion
if "working_ion" in X.columns:
    X = pd.get_dummies(
        X,
        columns=["working_ion"],
        prefix="ion",
        dtype=int
    )

# Combine target
final_df = pd.concat([X, y], axis=1)

# Save
final_df.to_csv(OUTPUT_FILE, index=False)

print("\nFinal shape:")
print(final_df.shape)

print("\nTarget:")
print(TARGET)

print("\nNumber of input features:")
print(final_df.shape[1] - 1)

print("\nMissing values:")
print(final_df.isnull().sum().sum())

print("\nNon-numeric columns:")
print(
    final_df.select_dtypes(
        exclude="number"
    ).columns.tolist()
)

print("\nWorking-ion categories:")
print(
    [
        col for col in X.columns
        if col.startswith("ion_")
    ]
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("MIXED DATASET PREPARATION COMPLETED")
print("=" * 60)