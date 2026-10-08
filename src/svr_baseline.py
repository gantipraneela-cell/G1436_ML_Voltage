import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# LOAD PCA DATA
# ============================================

X_train = pd.read_csv("X_train_pca.csv")
X_test = pd.read_csv("X_test_pca.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("====================================")
print("DATA LOADING")
print("====================================")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================
# SVR MODEL
# ============================================

print("\n====================================")
print("TRAINING SVR")
print("====================================")

model = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)

model.fit(X_train, y_train)

print("SVR training completed.")


# ============================================
# PREDICTION
# ============================================

y_pred = model.predict(X_test)


# ============================================
# EVALUATION
# ============================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

accuracy_05 = np.mean(
    np.abs(y_test - y_pred) <= 0.5
) * 100


# ============================================
# DISPLAY RESULTS
# ============================================

print("\n====================================")
print("SVR BASELINE RESULTS")
print("====================================")

print(f"MAE              : {mae:.4f} V")
print(f"RMSE             : {rmse:.4f} V")
print(f"R²               : {r2:.4f}")
print(f"±0.5 V Accuracy  : {accuracy_05:.2f}%")


# ============================================
# SAVE PREDICTIONS
# ============================================

results = pd.DataFrame({
    "Actual_Voltage": y_test.values,
    "Predicted_Voltage": y_pred,
    "Absolute_Error": np.abs(y_test.values - y_pred)
})

results.to_csv(
    "svr_predictions.csv",
    index=False
)

print("\nPredictions saved successfully:")
print("svr_predictions.csv")