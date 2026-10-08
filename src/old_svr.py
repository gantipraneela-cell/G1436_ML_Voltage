import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ==========================================
# LOAD OLD 90/10 PCA DATA
# ==========================================

base = "../data/preprocessed/"

X_train = pd.read_csv(base + "X_train_pca.csv")
X_test = pd.read_csv(base + "X_test_pca.csv")

y_train = pd.read_csv(base + "y_train.csv").squeeze()
y_test = pd.read_csv(base + "y_test.csv").squeeze()

print("OLD SVR EXPERIMENT")
print("==================")

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

# ==========================================
# SVR MODEL
# ==========================================

model = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)

print("\nTraining SVR...")

model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

accuracy = np.mean(np.abs(y_test - y_pred) <= 0.5) * 100

print("\nRESULTS")
print("=======")

print(f"MAE              : {mae:.4f} V")
print(f"RMSE             : {rmse:.4f} V")
print(f"R²               : {r2:.4f}")
print(f"±0.5 V Accuracy  : {accuracy:.2f}%")

# ==========================================
# SAVE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Absolute_Error": np.abs(y_test - y_pred)
})

results.to_csv("old_svr_predictions.csv", index=False)

print("\nPredictions saved to:")
print("old_svr_predictions.csv")