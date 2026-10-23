
import os
import numpy as np
import pandas as pd

# ============================================================
# 1. FILE PATHS AND COLUMN NAMES
# ============================================================

PREDICTIONS_PATH = (
    "data/preprocessed/li_svr_hgb_final_test_predictions.csv"
)

OUTPUT_PATH = (
    "data/preprocessed/li_svr_hgb_error_analysis.csv"
)

actual_col = "actual_voltage"
prediction_col = "hybrid_prediction"
group_col = "battery_formula"


# ============================================================
# 2. LOAD THE SAVED TEST PREDICTIONS
# ============================================================

if not os.path.exists(PREDICTIONS_PATH):
    raise FileNotFoundError(
        f"Prediction file not found: {PREDICTIONS_PATH}"
    )

df = pd.read_csv(PREDICTIONS_PATH)

required_columns = [
    group_col,
    actual_col,
    prediction_col,
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df[actual_col] = pd.to_numeric(
    df[actual_col], errors="coerce"
)

df[prediction_col] = pd.to_numeric(
    df[prediction_col], errors="coerce"
)

# Remove rows where actual or predicted voltage is unavailable.
df = df.dropna(
    subset=[actual_col, prediction_col]
).copy()

print("=" * 65)
print("SVR + HGB HYBRID: TEST ERROR ANALYSIS")
print("=" * 65)

print("Prediction file:", PREDICTIONS_PATH)
print("Valid test samples:", len(df))


# ============================================================
# 3. CALCULATE PREDICTION ERRORS
# ============================================================

# Positive error: model predicted too high.
# Negative error: model predicted too low.

df["Error_V"] = (
    df[prediction_col] - df[actual_col]
)

df["Absolute_Error_V"] = df["Error_V"].abs()

df["Squared_Error_V"] = (
    df["Error_V"] ** 2
)

# Sort from largest error to smallest error.
df_sorted = df.sort_values(
    "Absolute_Error_V",
    ascending=False,
).copy()


# ============================================================
# 4. CLASSIFY PREDICTION ACCURACY
# ============================================================

df_sorted["Accuracy_Category"] = np.select(
    [
        df_sorted["Absolute_Error_V"] <= 0.25,
        df_sorted["Absolute_Error_V"] <= 0.50,
        df_sorted["Absolute_Error_V"] <= 1.00,
    ],
    [
        "Excellent: within 0.25 V",
        "Good: within 0.50 V",
        "Moderate: within 1.00 V",
    ],
    default="Large error: above 1.00 V",
)


# ============================================================
# 5. GROUP SAMPLES BY ACTUAL VOLTAGE
# ============================================================

df_sorted["Voltage_Range"] = pd.cut(
    df_sorted[actual_col],
    bins=[-np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf],
    labels=[
        "Below 1 V",
        "1 to below 2 V",
        "2 to below 3 V",
        "3 to below 4 V",
        "4 to 5 V",
        "Above 5 V",
    ],
    right=False,
)

# Note: intervals are left-inclusive and right-exclusive.
# Thus, 1.0 V belongs to "1 to below 2 V", and
# 5.0 V belongs to "Above 5 V" with these bins.


# ============================================================
# 6. OVERALL ERROR ANALYSIS
# ============================================================

actual = df_sorted[actual_col]
predicted = df_sorted[prediction_col]
absolute_error = df_sorted["Absolute_Error_V"]
signed_error = df_sorted["Error_V"]

print("\n" + "=" * 65)
print("OVERALL ERROR ANALYSIS")
print("=" * 65)

print("Total test samples:", len(df_sorted))

print(
    "MAE:",
    f"{absolute_error.mean():.4f} V",
)

print(
    "RMSE:",
    f"{np.sqrt(df_sorted['Squared_Error_V'].mean()):.4f} V",
)

print(
    "Mean signed error (bias):",
    f"{signed_error.mean():.4f} V",
)

print(
    "Median absolute error:",
    f"{absolute_error.median():.4f} V",
)

print(
    "Within 0.25 V:",
    f"{(absolute_error <= 0.25).mean() * 100:.2f}%",
)

print(
    "Within 0.50 V:",
    f"{(absolute_error <= 0.50).mean() * 100:.2f}%",
)

print(
    "Within 1.00 V:",
    f"{(absolute_error <= 1.00).mean() * 100:.2f}%",
)

print(
    "Errors above 1.00 V:",
    int((absolute_error > 1.00).sum()),
)

print(
    "Predictions too high:",
    int((signed_error > 0).sum()),
)

print(
    "Predictions too low:",
    int((signed_error < 0).sum()),
)


# ============================================================
# 7. TOP 20 LARGEST PREDICTION ERRORS
# ============================================================

print("\n" + "=" * 65)
print("TOP 20 LARGEST PREDICTION ERRORS")
print("=" * 65)

columns_to_show = [
    group_col,
    actual_col,
    prediction_col,
    "Error_V",
    "Absolute_Error_V",
    "Voltage_Range",
    "Accuracy_Category",
]

print(
    df_sorted[columns_to_show]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# ============================================================
# 8. ANALYZE ACTUAL VOLTAGES BELOW 1 V
# ============================================================

low_voltage = df_sorted[
    df_sorted[actual_col] < 1.0
].copy()

print("\n" + "=" * 65)
print("ACTUAL VOLTAGE BELOW 1 V")
print("=" * 65)

print("Number of samples:", len(low_voltage))

if len(low_voltage) > 0:

    print(
        "MAE:",
        f"{low_voltage['Absolute_Error_V'].mean():.4f} V",
    )

    print(
        "Mean signed error:",
        f"{low_voltage['Error_V'].mean():.4f} V",
    )

    print(
        "Within 0.50 V:",
        f"{(low_voltage['Absolute_Error_V'] <= 0.50).mean() * 100:.2f}%",
    )

    print(
        low_voltage[columns_to_show]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

else:
    print("No test samples below 1 V.")


# ============================================================
# 9. ANALYZE ALL VOLTAGE RANGES
# ============================================================

print("\n" + "=" * 65)
print("ERROR BY ACTUAL VOLTAGE RANGE")
print("=" * 65)

range_summary = (
    df_sorted.groupby(
        "Voltage_Range",
        observed=False,
    )
    .agg(
        Samples=(actual_col, "size"),
        MAE_V=("Absolute_Error_V", "mean"),
        Median_Absolute_Error_V=(
            "Absolute_Error_V", "median"
        ),
        Mean_Signed_Error_V=("Error_V", "mean"),
        Within_0_5V_Percent=(
            "Absolute_Error_V",
            lambda values: (
                values.le(0.5).mean() * 100
            ),
        ),
    )
    .reset_index()
)

print(
    range_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# ============================================================
# 10. ANALYZE ERRORS BY MODEL COMPONENT
# ============================================================

component_columns = {
    "SVR": "svr_prediction",
    "Standard HGB": "hgb_prediction",
    "Weighted HGB": "weighted_hgb_prediction",
}

print("\n" + "=" * 65)
print("MODEL COMPONENT COMPARISON ON THE TEST SET")
print("=" * 65)

component_results = []

for model_name, column in component_columns.items():

    if column not in df.columns:
        print(
            f"Skipping {model_name}: "
            f"column '{column}' is unavailable."
        )
        continue

    component_actual = df[actual_col]
    component_predicted = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    valid = (
        component_actual.notna()
        & component_predicted.notna()
    )

    if not valid.any():
        continue

    errors = (
        component_predicted[valid]
        - component_actual[valid]
    )

    abs_errors = errors.abs()

    result = {
        "Model": model_name,
        "Samples": int(valid.sum()),
        "MAE_V": abs_errors.mean(),
        "RMSE_V": np.sqrt((errors ** 2).mean()),
        "Within_0_5V_Percent": (
            abs_errors.le(0.5).mean() * 100
        ),
    }

    component_results.append(result)

if component_results:

    component_summary = pd.DataFrame(
        component_results
    )

    print(
        component_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    component_summary.to_csv(
        "data/preprocessed/"
        "li_svr_hgb_component_error_summary.csv",
        index=False,
    )

    print(
        "\nComponent comparison saved to:"
        "\ndata/preprocessed/"
        "li_svr_hgb_component_error_summary.csv"
    )


# ============================================================
# 11. SAVE DETAILED ERROR ANALYSIS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

df_sorted.to_csv(
    OUTPUT_PATH,
    index=False,
)

range_summary.to_csv(
    "data/preprocessed/"
    "li_svr_hgb_voltage_range_error_summary.csv",
    index=False,
)

print("\n" + "=" * 65)
print("FILES SAVED SUCCESSFULLY")
print("=" * 65)

print("Detailed errors:")
print(OUTPUT_PATH)

print(
    "Voltage-range summary:"
    "\ndata/preprocessed/"
    "li_svr_hgb_voltage_range_error_summary.csv"
)

print("\nAnalysis completed.")
