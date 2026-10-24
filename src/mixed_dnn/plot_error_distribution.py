import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("STEP 6B - PREDICTION ERROR ANALYSIS")
print("=" * 70)

# Load predictions
df = pd.read_csv("results/mixed_dnn/E2_test_predictions.csv")

actual = df["actual_average_voltage"].values
predicted = df["predicted_average_voltage"].values

# Calculate errors
error = predicted - actual
absolute_error = np.abs(error)

print("\nNumber of test samples:", len(df))

print("\nError statistics:")
print(f"Mean Error              : {error.mean():.6f}")
print(f"Mean Absolute Error     : {absolute_error.mean():.6f}")
print(f"Median Absolute Error   : {np.median(absolute_error):.6f}")
print(f"Maximum Absolute Error  : {absolute_error.max():.6f}")
print(f"Minimum Error           : {error.min():.6f}")
print(f"Maximum Error           : {error.max():.6f}")

# Find largest errors
largest_errors = np.argsort(absolute_error)[-10:][::-1]

print("\nTop 10 largest prediction errors:")
print("-" * 70)
print(f"{'Actual':>12} {'Predicted':>12} {'Error':>12} {'Abs Error':>12}")
print("-" * 70)

for i in largest_errors:
    print(
        f"{actual[i]:12.6f} "
        f"{predicted[i]:12.6f} "
        f"{error[i]:12.6f} "
        f"{absolute_error[i]:12.6f}"
    )

# Plot error distribution
plt.figure(figsize=(8, 6))

plt.hist(error, bins=30, alpha=0.7)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Number of Samples")
plt.title("DNN - Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    "results/mixed_dnn/prediction_error_distribution.png",
    dpi=300
)

plt.show()

print("\nPlot saved as:")
print("results/mixed_dnn/prediction_error_distribution.png")

print("\n" + "=" * 70)
print("STEP 6B COMPLETED")
print("=" * 70)