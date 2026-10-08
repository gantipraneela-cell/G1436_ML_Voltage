import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# LOAD SVR PREDICTIONS
# ============================================

df = pd.read_csv("svr_predictions.csv")

actual = df["Actual_Voltage"]
predicted = df["Predicted_Voltage"]

# ============================================
# CREATE VOLTAGE RANGES
# ============================================

bins = [-np.inf, 2, 3, 4, 5, np.inf]

labels = [
    "Below 2 V",
    "2–3 V",
    "3–4 V",
    "4–5 V",
    "Above 5 V"
]

df["Voltage_Range"] = pd.cut(
    actual,
    bins=bins,
    labels=labels
)

# ============================================
# ERROR ANALYSIS
# ============================================

print("============================================")
print("SVR ERROR ANALYSIS BY VOLTAGE RANGE")
print("============================================")

results = []

for voltage_range in labels:

    group = df[df["Voltage_Range"] == voltage_range]

    y_true = group["Actual_Voltage"]
    y_pred = group["Predicted_Voltage"]

    if len(group) == 0:
        continue

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    accuracy_05 = np.mean(
        np.abs(y_true - y_pred) <= 0.5
    ) * 100

    if len(group) > 1:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    print("\n" + voltage_range)
    print("--------------------------------------------")
    print(f"Samples          : {len(group)}")
    print(f"MAE              : {mae:.4f} V")
    print(f"RMSE             : {rmse:.4f} V")
    print(f"±0.5 V Accuracy  : {accuracy_05:.2f}%")
    print(f"R²               : {r2:.4f}")

    results.append({
        "Voltage_Range": voltage_range,
        "Samples": len(group),
        "MAE": mae,
        "RMSE": rmse,
        "Accuracy_±0.5V": accuracy_05,
        "R2": r2
    })

# ============================================
# SAVE RESULTS
# ============================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "svr_error_analysis.csv",
    index=False
)

print("\n============================================")
print("ERROR ANALYSIS COMPLETED")
print("============================================")

print("\nResults saved as:")
print("svr_error_analysis.csv")