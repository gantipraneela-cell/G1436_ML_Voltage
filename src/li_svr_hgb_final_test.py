
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# CONFIGURATION
# ==================================================

TRAIN_PATH = "data/preprocessed/li_v2_train_composition.csv"
TEST_PATH = "data/preprocessed/li_v2_test_composition.csv"

RESULTS_PATH = "data/preprocessed/li_svr_hgb_final_test_results.csv"
PREDICTIONS_PATH = "data/preprocessed/li_svr_hgb_final_test_predictions.csv"

TARGET = "average_voltage"
GROUP = "battery_formula"

# Representative settings from nested grouped CV
SVR_WEIGHT = 0.7
HGB_WEIGHT = 0.3
LOW_VOLTAGE_THRESHOLD = 2.0


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(y_true, predictions):
    errors = np.abs(y_true - predictions)
    low = y_true < 1.0

    return {
        "MAE_V": mean_absolute_error(y_true, predictions),
        "RMSE_V": np.sqrt(
            mean_squared_error(y_true, predictions)
        ),
        "R2": r2_score(y_true, predictions),
        "within_0.5V_percent": np.mean(errors <= 0.5) * 100,
        "below_1V_count": int(np.sum(low)),
        "below_1V_MAE": (
            mean_absolute_error(y_true[low], predictions[low])
            if low.any() else np.nan
        ),
        "below_1V_within_0.5V_percent": (
            np.mean(errors[low] <= 0.5) * 100
            if low.any() else np.nan
        )
    }


# ==================================================
# LOAD DATA
# ==================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for name, frame in [
    ("Training", train_df),
    ("Test", test_df)
]:
    if TARGET not in frame.columns:
        raise ValueError(
            f"{TARGET} missing from {name.lower()} data."
        )

    if GROUP not in frame.columns:
        raise ValueError(
            f"{GROUP} missing from {name.lower()} data."
        )

y_train = train_df[TARGET].astype(float).to_numpy()
y_test = test_df[TARGET].astype(float).to_numpy()

if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
    raise ValueError("Target contains missing or infinite values.")

train_groups = train_df[GROUP].astype(str).to_numpy()
test_groups = test_df[GROUP].astype(str).to_numpy()

# Check for formula overlap between training and test sets.
overlap = set(train_groups).intersection(set(test_groups))

print("=" * 75)
print("LI-ONLY SVR + HGB: FINAL HELD-OUT TEST EVALUATION")
print("=" * 75)

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
print("Training unique formulas:", len(set(train_groups)))
print("Test unique formulas:", len(set(test_groups)))
print("Formulas shared across train and test:", len(overlap))

if overlap:
    raise ValueError(
        "Formula overlap detected. Stop and inspect the grouped split "
        "before reporting this as an unseen-formula test."
    )


# ==================================================
# PREPARE FEATURES
# ==================================================

X_train = train_df.drop(columns=[TARGET, GROUP])
X_test = test_df.drop(columns=[TARGET, GROUP])

X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Use only features present in both datasets, preserving
# the training feature order.
common_features = [
    column for column in X_train.columns
    if column in X_test.columns
]

if not common_features:
    raise ValueError("No common numeric features found.")

X_train = X_train[common_features].copy()
X_test = X_test[common_features].copy()

# Remove features that are constant in training data.
constant_features = [
    column for column in X_train.columns
    if X_train[column].nunique(dropna=True) <= 1
]

X_train.drop(columns=constant_features, inplace=True)
X_test.drop(columns=constant_features, inplace=True)

if X_train.shape[1] == 0:
    raise ValueError("No usable features remain.")

print("\nFeatures used:", X_train.shape[1])
print("Constant features removed:", len(constant_features))

# SVR and standard HGB weights:
# 1.5x for training samples below 2 V.
train_weights = np.where(y_train < 2.0, 1.5, 1.0)

# Stronger low-voltage HGB weighting:
# 5x below 1 V, 1.5x from 1 V to below 2 V.
low_hgb_weights = np.where(y_train < 2.0, 1.5, 1.0)
low_hgb_weights[y_train < 1.0] = 5.0


# ==================================================
# MODEL 1: SVR
# ==================================================

print("\nTraining SVR...")

svr = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    SVR(
        kernel="rbf",
        C=30,
        gamma=0.005,
        epsilon=0.25
    )
)

svr.fit(
    X_train,
    y_train,
    svr__sample_weight=train_weights
)

pred_svr = svr.predict(X_test)


# ==================================================
# MODEL 2: STANDARD HGB
# ==================================================

print("Training standard HGB...")

hgb = make_pipeline(
    SimpleImputer(strategy="median"),
    HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
)

hgb.fit(
    X_train,
    y_train,
    histgradientboostingregressor__sample_weight=train_weights
)

pred_hgb = hgb.predict(X_test)


# ==================================================
# MODEL 3: LOW-VOLTAGE-WEIGHTED HGB
# ==================================================

print("Training low-voltage-weighted HGB...")

hgb_low = make_pipeline(
    SimpleImputer(strategy="median"),
    HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
)

hgb_low.fit(
    X_train,
    y_train,
    histgradientboostingregressor__sample_weight=low_hgb_weights
)

pred_hgb_low = hgb_low.predict(X_test)


# ==================================================
# MODEL 4: HYBRID
# ==================================================

print("Creating hybrid predictions...")

# Ordinary blend: 70% SVR + 30% standard HGB.
ordinary_blend = (
    SVR_WEIGHT * pred_svr
    + HGB_WEIGHT * pred_hgb
)

# Replace the ordinary blend with low-voltage HGB
# when its prediction is below 2 V.
pred_hybrid = np.where(
    pred_hgb_low < LOW_VOLTAGE_THRESHOLD,
    pred_hgb_low,
    ordinary_blend
)


# ==================================================
# EVALUATE ALL MODELS ON THE SAME TEST ROWS
# ==================================================

models = [
    ("SVR_only", pred_svr),
    ("HGB_standard", pred_hgb),
    ("HGB_low_voltage_weight_5x", pred_hgb_low),
    ("Hybrid_70_30_threshold_2V", pred_hybrid)
]

results = []

for model_name, predictions in models:
    row = {
        "model": model_name,
        **calculate_metrics(y_test, predictions)
    }
    results.append(row)

results_df = pd.DataFrame(results)


# ==================================================
# VOLTAGE-RANGE ERROR ANALYSIS
# ==================================================

range_rows = []

voltage_ranges = [
    ("Below 1 V", y_test < 1.0),
    ("1 to below 2 V", (y_test >= 1.0) & (y_test < 2.0)),
    ("2 to below 3 V", (y_test >= 2.0) & (y_test < 3.0)),
    ("3 to below 4 V", (y_test >= 3.0) & (y_test < 4.0)),
    ("4 to 5 V", (y_test >= 4.0) & (y_test <= 5.0)),
    ("Above 5 V", y_test > 5.0)
]

for range_name, mask in voltage_ranges:
    if not mask.any():
        continue

    for model_name, predictions in models:
        errors = np.abs(y_test[mask] - predictions[mask])

        range_rows.append({
            "voltage_range": range_name,
            "model": model_name,
            "sample_count": int(mask.sum()),
            "MAE_V": mean_absolute_error(
                y_test[mask],
                predictions[mask]
            ),
            "within_0.5V_percent": np.mean(errors <= 0.5) * 100
        })

range_df = pd.DataFrame(range_rows)


# ==================================================
# SAVE RESULTS
# ==================================================

os.makedirs("data/preprocessed", exist_ok=True)

results_df.to_csv(RESULTS_PATH, index=False)

predictions_df = pd.DataFrame({
    "battery_formula": test_groups,
    "actual_voltage": y_test,
    "svr_prediction": pred_svr,
    "hgb_prediction": pred_hgb,
    "weighted_hgb_prediction": pred_hgb_low,
    "hybrid_prediction": pred_hybrid
})

predictions_df["hybrid_absolute_error"] = np.abs(
    y_test - pred_hybrid
)

predictions_df.to_csv(PREDICTIONS_PATH, index=False)

range_path = (
    "data/preprocessed/li_svr_hgb_final_test_voltage_ranges.csv"
)

range_df.to_csv(range_path, index=False)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 100)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\n" + "=" * 100)
print("ERROR ANALYSIS BY ACTUAL VOLTAGE RANGE")
print("=" * 100)

print(
    range_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nFiles saved:")
print("Results:", RESULTS_PATH)
print("Predictions:", PREDICTIONS_PATH)
print("Voltage-range analysis:", range_path)

print("\nEvaluation completed.")
print("The held-out test data was not used to fit the models.")
print("Do not tune the model based on these test results and")
print("then report the same test score as an unbiased final estimate.")
