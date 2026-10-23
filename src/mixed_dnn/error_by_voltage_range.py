import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 6D - ERROR BY VOLTAGE RANGE")
print("=" * 70)

# Load predictions
df = pd.read_csv("results/mixed_dnn/E2_test_predictions.csv")

actual = df["actual_average_voltage"].values
predicted = df["predicted_average_voltage"].values

error = predicted - actual
absolute_error = np.abs(error)

# Create voltage ranges
bins = [-np.inf, 0, 2, 4, 6, 8, np.inf]
labels = [
    "< 0",
    "0 - 2",
    "2 - 4",
    "4 - 6",
    "6 - 8",
    "> 8"
]

df["Voltage_Range"] = pd.cut(
    actual,
    bins=bins,
    labels=labels
)

df["Error"] = error
df["Absolute_Error"] = absolute_error

# Calculate statistics by range
summary = df.groupby(
    "Voltage_Range",
    observed=False
).agg(
    Samples=("actual_average_voltage", "count"),
    Mean_Error=("Error", "mean"),
    MAE=("Absolute_Error", "mean"),
    Maximum_Absolute_Error=("Absolute_Error", "max")
)

print("\nERROR PERFORMANCE BY ACTUAL VOLTAGE RANGE")
print("=" * 70)

print(summary.to_string(float_format=lambda x: f"{x:.6f}"))

# Save results
summary.to_csv("results/mixed_dnn/E2_error_by_voltage_range.csv", index=False)

print("\nSaved:")
print("E2_error_by_voltage_range.csv")

print("\n" + "=" * 70)
print("STEP 6D COMPLETED")
print("=" * 70)