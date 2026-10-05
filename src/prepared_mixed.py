import pandas as pd

print("=" * 60)
print("FINAL MIXED DATASET PREPARATION")
print("=" * 60)

input_file = "battery_mixed_feature_engineered.csv"
output_file = "battery_mixed_final.csv"

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(input_file)

print("\nOriginal shape:")
print(df.shape)

# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

target = "average_voltage"

# ------------------------------------------------------------
# 3. REMOVE EXACTLY REDUNDANT FEATURE
# ------------------------------------------------------------

if "total_atoms" in df.columns:
    df = df.drop(columns=["total_atoms"])
    print("\nRemoved:")
    print("total_atoms")

# Keep num_sites because total_atoms and num_sites
# contain identical information in this dataset.

# ------------------------------------------------------------
# 4. ENCODE WORKING ION
# ------------------------------------------------------------

if "working_ion" in df.columns:

    print("\nWorking ion categories:")
    print(sorted(df["working_ion"].dropna().unique()))

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=["working_ion"],
        prefix="ion",
        dtype=int
    )

    print("\nworking_ion encoded using one-hot encoding.")

# ------------------------------------------------------------
# 5. CHECK MISSING VALUES
# ------------------------------------------------------------

missing = df.isnull().sum().sum()

print("\nTotal missing values:")
print(missing)

# ------------------------------------------------------------
# 6. CHECK DATA TYPES
# ------------------------------------------------------------

non_numeric = df.drop(columns=[target]).select_dtypes(
    exclude="number"
).columns.tolist()

print("\nRemaining non-numeric features:")
print(non_numeric)

# ------------------------------------------------------------
# 7. FINAL SHAPE
# ------------------------------------------------------------

print("\nFinal shape:")
print(df.shape)

print("\nFinal number of input features:")
print(df.shape[1] - 1)

# ------------------------------------------------------------
# 8. SAVE
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 60)
print("MIXED DATASET PREPARATION COMPLETED")
print("=" * 60)