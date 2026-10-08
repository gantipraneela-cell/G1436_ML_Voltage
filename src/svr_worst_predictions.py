import pandas as pd
import numpy as np

# ============================================
# LOAD SVR PREDICTIONS
# ============================================

df = pd.read_csv("svr_predictions.csv")

# ============================================
# CALCULATE ERRORS
# ============================================

df["Error"] = (
    df["Predicted_Voltage"] - df["Actual_Voltage"]
)

df["Absolute_Error"] = np.abs(df["Error"])

# ============================================
# CREATE VOLTAGE RANGE
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
    df["Actual_Voltage"],
    bins=bins,
    labels=labels
)

# ============================================
# SORT BY LARGEST ERROR
# ============================================

worst = df.sort_values(
    by="Absolute_Error",
    ascending=False
).head(20)

# ============================================
# DISPLAY RESULTS
# ============================================

print("============================================")
print("20 WORST SVR PREDICTIONS")
print("============================================")

print(
    worst[
        [
            "Actual_Voltage",
            "Predicted_Voltage",
            "Error",
            "Absolute_Error",
            "Voltage_Range"
        ]
    ].to_string(index=False)
)

# ============================================
# SAVE RESULTS
# ============================================

worst.to_csv(
    "svr_worst_20_predictions.csv",
    index=False
)

print("\n============================================")
print("WORST PREDICTIONS ANALYSIS COMPLETED")
print("============================================")

print("\nSaved as:")
print("svr_worst_20_predictions.csv")