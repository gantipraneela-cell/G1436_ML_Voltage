import pandas as pd
import numpy as np
import os

print("=" * 60)
print("BATTERY DATA PREPROCESSING")
print("=" * 60)

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

files = {
    "Li": "battery_Li_data.csv",
    "Mixed": "battery_mixed_data.csv"
}

# ---------------------------------------------------------
# PROCESS EACH DATASET
# ---------------------------------------------------------

for name, filename in files.items():

    print("\n" + "=" * 60)
    print(f"PROCESSING {name} DATASET")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(filename)

    print("\nOriginal shape:")
    print(df.shape)

    # -----------------------------------------------------
    # 1. Remove duplicate rows
    # -----------------------------------------------------

    duplicates = df.duplicated().sum()

    print("\nDuplicate rows:", duplicates)

    df = df.drop_duplicates()

    # -----------------------------------------------------
    # 2. Check missing values
    # -----------------------------------------------------

    missing = df.isnull().sum().sum()

    print("Total missing values:", missing)

    # -----------------------------------------------------
    # 3. Remove target leakage columns
    # -----------------------------------------------------

    leakage_columns = [
        "energy_grav",
        "energy_vol"
    ]

    removed_leakage = []

    for col in leakage_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_leakage.append(col)

    print("\nRemoved leakage columns:")
    print(removed_leakage)

    # -----------------------------------------------------
    # 4. Remove constant columns
    # -----------------------------------------------------

    constant_columns = []

    for col in df.columns:

        if col == "average_voltage":
            continue

        if df[col].nunique(dropna=False) <= 1:
            constant_columns.append(col)

    df = df.drop(columns=constant_columns)

    print("\nConstant columns removed:")
    print(len(constant_columns))

    if constant_columns:
        print(constant_columns)

    # -----------------------------------------------------
    # 5. Check missing values again
    # -----------------------------------------------------

    missing_after = df.isnull().sum().sum()

    print("\nMissing values after cleaning:")
    print(missing_after)

    # -----------------------------------------------------
    # 6. Save cleaned dataset
    # -----------------------------------------------------

    output_file = f"battery_{name}_cleaned.csv"

    df.to_csv(output_file, index=False)

    print("\nCleaned dataset saved as:")
    print(output_file)

    print("\nFinal shape:")
    print(df.shape)

    # -----------------------------------------------------
    # 7. Target statistics
    # -----------------------------------------------------

    print("\nAverage voltage statistics:")

    print("Minimum:",
          df["average_voltage"].min())

    print("Maximum:",
          df["average_voltage"].max())

    print("Mean:",
          df["average_voltage"].mean())

    print("Median:",
          df["average_voltage"].median())

    # -----------------------------------------------------
    # 8. Working ion information
    # -----------------------------------------------------

    if "working_ion" in df.columns:

        print("\nWorking ion counts:")

        print(
            df["working_ion"].value_counts()
        )

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)