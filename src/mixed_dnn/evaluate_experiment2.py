import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/preprocessed/mixed_dnn")
MODEL_FILE = Path("models/mixed_dnn/experiment2_DNN_model.keras")
RESULTS_DIR = Path("results/mixed_dnn")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DNN EXPERIMENT 2 - TEST EVALUATION")
print("=" * 70)

# Load trained model
model = tf.keras.models.load_model(MODEL_FILE)

# Load untouched test data
X_test = pd.read_csv(DATA_DIR / "E2_X_test_pca.csv")
y_test = pd.read_csv(DATA_DIR / "E2_y_test.csv")

print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Convert to NumPy
X_test = X_test.astype(np.float32).values
y_test = y_test.astype(np.float32).values.ravel()

# Predict
y_pred = model.predict(X_test, verbose=0).ravel()

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("EXPERIMENT 2 TEST RESULTS")
print("=" * 70)

print(f"\nMAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

print(f"\nVariance explained (%): {r2 * 100:.2f}%")

print("\nPrediction range:")
print(f"Actual minimum    : {y_test.min():.6f}")
print(f"Actual maximum    : {y_test.max():.6f}")
print(f"Predicted minimum : {y_pred.min():.6f}")
print(f"Predicted maximum : {y_pred.max():.6f}")

# Save predictions
results = pd.DataFrame({
    "actual_average_voltage": y_test,
    "predicted_average_voltage": y_pred,
    "error": y_pred - y_test
})

results.to_csv(RESULTS_DIR / "E2_test_predictions.csv", index=False)

# Save metrics
metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "Variance Explained (%)"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2,
        r2 * 100
    ]
})

metrics.to_csv(RESULTS_DIR / "E2_test_metrics.csv", index=False)

print("\nFiles saved:")
print(RESULTS_DIR / "E2_test_predictions.csv")
print(RESULTS_DIR / "E2_test_metrics.csv")

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)