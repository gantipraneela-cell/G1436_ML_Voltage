import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("STEP 6C - ERROR VS ACTUAL VOLTAGE")
print("=" * 70)

# Load predictions
df = pd.read_csv("results/mixed_dnn/E2_test_predictions.csv")

actual = df["actual_average_voltage"].values
predicted = df["predicted_average_voltage"].values

# Calculate error
error = predicted - actual

print("\nNumber of test samples:", len(df))

print(f"Mean error: {error.mean():.6f}")
print(f"Minimum error: {error.min():.6f}")
print(f"Maximum error: {error.max():.6f}")

# Create plot
plt.figure(figsize=(8, 6))

plt.scatter(actual, error, alpha=0.6)

# Zero-error reference line
plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Actual Average Voltage")
plt.ylabel("Prediction Error (Predicted - Actual)")
plt.title("DNN - Prediction Error vs Actual Average Voltage")

plt.tight_layout()

plt.savefig(
    "results/mixed_dnn/error_vs_actual.png",
    dpi=300
)

plt.show()

print("\nPlot saved as:")
print("results/mixed_dnn/error_vs_actual.png")

print("\n" + "=" * 70)
print("STEP 6C COMPLETED")
print("=" * 70)