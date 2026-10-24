import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("STEP 6A - ACTUAL VS PREDICTED ANALYSIS")
print("=" * 70)

# Load test predictions
df = pd.read_csv("results/mixed_dnn/E2_test_predictions.csv")

# Get actual and predicted values
actual = df["actual_average_voltage"]
predicted = df["predicted_average_voltage"]

print("\nNumber of test samples:", len(df))

print("\nActual voltage:")
print(f"Minimum : {actual.min():.6f}")
print(f"Maximum : {actual.max():.6f}")
print(f"Mean    : {actual.mean():.6f}")

print("\nPredicted voltage:")
print(f"Minimum : {predicted.min():.6f}")
print(f"Maximum : {predicted.max():.6f}")
print(f"Mean    : {predicted.mean():.6f}")

# Create actual vs predicted plot
plt.figure(figsize=(8, 6))

plt.scatter(actual, predicted, alpha=0.6)

# Perfect prediction line
min_value = min(actual.min(), predicted.min())
max_value = max(actual.max(), predicted.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Average Voltage")
plt.ylabel("Predicted Average Voltage")
plt.title("DNN - Actual vs Predicted Average Voltage")

plt.tight_layout()

# Save figure
plt.savefig("results/mixed_dnn/actual_vs_predicted.png", dpi=300)

plt.show()

print("\nPlot saved as:")
print("actual_vs_predicted.png")

print("\n" + "=" * 70)
print("STEP 6A COMPLETED")
print("=" * 70)