import pandas as pd

print("=" * 60)
print("DATASET CLEANING")
print("=" * 60)

# ============================================================
# LI DATASET
# ============================================================

li_input = "battery_Li_data.csv"
li_output = "battery_Li_cleaned.csv"

df_li = pd.read_csv(li_input)

print("\nLi dataset before cleaning:")
print(df_li.shape)

# Remove duplicate rows
df_li = df_li.drop_duplicates()

# Remove leakage / unnecessary columns
columns_to_drop = [
    "energy_grav",
    "energy_vol"
]

for col in columns_to_drop:
    if col in df_li.columns:
        df_li = df_li.drop(columns=col)

# Remove columns containing only one unique value
constant_columns = []

for col in df_li.columns:
    if col != "average_voltage":
        if df_li[col].nunique(dropna=False) <= 1:
            constant_columns.append(col)

if constant_columns:
    df_li = df_li.drop(columns=constant_columns)

# Save cleaned Li dataset
df_li.to_csv(li_output, index=False)

print("\nLi dataset after cleaning:")
print(df_li.shape)

print("\nRemoved constant columns:")
print(constant_columns)

print("\nMissing values:")
print(df_li.isnull().sum().sum())

print("\nSaved:")
print(li_output)


# ============================================================
# MIXED DATASET
# ============================================================

mixed_input = "battery_mixed_data.csv"
mixed_output = "battery_mixed_cleaned.csv"

df_mixed = pd.read_csv(mixed_input)

print("\n" + "=" * 60)
print("MIXED DATASET")
print("=" * 60)

print("\nMixed dataset before cleaning:")
print(df_mixed.shape)

# Remove duplicate rows
df_mixed = df_mixed.drop_duplicates()

# Remove leakage columns
columns_to_drop = [
    "energy_grav",
    "energy_vol"
]

for col in columns_to_drop:
    if col in df_mixed.columns:
        df_mixed = df_mixed.drop(columns=col)

# Remove constant columns
constant_columns = []

for col in df_mixed.columns:
    if col != "average_voltage":
        if df_mixed[col].nunique(dropna=False) <= 1:
            constant_columns.append(col)

if constant_columns:
    df_mixed = df_mixed.drop(columns=constant_columns)

# Save cleaned mixed dataset
df_mixed.to_csv(mixed_output, index=False)

print("\nMixed dataset after cleaning:")
print(df_mixed.shape)

print("\nRemoved constant columns:")
print(constant_columns)

print("\nMissing values:")
print(df_mixed.isnull().sum().sum())

print("\nSaved:")
print(mixed_output)

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)