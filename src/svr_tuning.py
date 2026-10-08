import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# LOAD DATA
# ============================================

X_train = pd.read_csv("X_train_pca.csv")
X_test = pd.read_csv("X_test_pca.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("============================================")
print("SVR HYPERPARAMETER TUNING")
print("============================================")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================
# PARAMETER GRID
# ============================================

param_grid = {
    "C": [1, 10, 100, 500, 1000],
    "gamma": ["scale", 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.1, 0.2, 0.5]
}


# ============================================
# CROSS-VALIDATION
# ============================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

svr = SVR(
    kernel="rbf"
)

grid_search = GridSearchCV(
    estimator=svr,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# ============================================
# TRAIN
# ============================================

print("\nStarting Grid Search...")

grid_search.fit(X_train, y_train)

print("\nGrid Search completed.")


# ============================================
# BEST PARAMETERS
# ============================================

print("\n============================================")
print("BEST SVR PARAMETERS")
print("============================================")

print("Best parameters:")
print(grid_search.best_params_)

print(
    f"Best CV MAE: "
    f"{-grid_search.best_score_:.4f} V"
)


# ============================================
# FINAL MODEL
# ============================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)


# ============================================
# TEST SET EVALUATION
# ============================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

accuracy_05 = np.mean(
    np.abs(y_test - y_pred) <= 0.5
) * 100


# ============================================
# RESULTS
# ============================================

print("\n============================================")
print("TUNED SVR TEST RESULTS")
print("============================================")

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
    "Absolute_Error": np.abs(
        y_test.values - y_pred
    )
})

results.to_csv(
    "svr_tuned_predictions.csv",
    index=False
)

print("\nPredictions saved as:")
print("svr_tuned_predictions.csv")