import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# Create directories to save outputs
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Preprocessed Data
# ---------------------------------------------------------
print("Loading preprocessed PCA data...")
X_train = pd.read_csv("preprocessing/mixed_X_train_pca.csv").values
X_test = pd.read_csv("preprocessing/mixed_X_test_pca.csv").values

# Load target voltage values (flatten to 1D arrays)
y_train = pd.read_csv("preprocessing/mixed_y_train.csv").values.ravel()
y_test = pd.read_csv("preprocessing/mixed_y_test.csv").values.ravel()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}\n")

# ---------------------------------------------------------
# 2. Train Deep Neural Network (DNN)
# ---------------------------------------------------------
print("--- Training Deep Neural Network (DNN) ---")
dnn_model = Sequential([
    Dense(60, activation="relu", kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    Dropout(0.25),
    Dense(30, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.10),
    Dense(1)
])

dnn_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

history = dnn_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1
)

dnn_model.save("models/dnn_model.keras")
y_pred_dnn = dnn_model.predict(X_test).ravel()
print("DNN Training Complete.\n")

# ---------------------------------------------------------
# 3. Train Support Vector Regression (SVR)
# ---------------------------------------------------------
print("--- Training Support Vector Regression (SVR) ---")
svr_model = SVR(kernel="rbf", C=10.0, gamma=0.1)
svr_model.fit(X_train, y_train)

joblib.dump(svr_model, "models/svr_model.pkl")
y_pred_svr = svr_model.predict(X_test)
print("SVR Training Complete.\n")

# ---------------------------------------------------------
# 4. Train Kernel Ridge Regression (KRR)
# ---------------------------------------------------------
print("--- Training Kernel Ridge Regression (KRR) ---")
krr_model = KernelRidge(kernel="rbf", alpha=0.01, gamma=0.1)
krr_model.fit(X_train, y_train)

joblib.dump(krr_model, "models/krr_model.pkl")
y_pred_krr = krr_model.predict(X_test)
print("KRR Training Complete.\n")

# ---------------------------------------------------------
# 5. Evaluate All Models
# ---------------------------------------------------------
def get_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE (V)": mae, "RMSE (V)": rmse, "R2 Score": r2}

results = [
    get_metrics(y_test, y_pred_dnn, "DNN"),
    get_metrics(y_test, y_pred_svr, "SVR"),
    get_metrics(y_test, y_pred_krr, "KRR")
]

results_df = pd.DataFrame(results)
print("==========================================")
print("           MODEL EVALUATION RESULTS       ")
print("==========================================")
print(results_df.to_string(index=False))
results_df.to_csv("results/evaluation_metrics.csv", index=False)

# ---------------------------------------------------------
# 6. Plot Actual vs Predicted Graphs
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
predictions = [("DNN", y_pred_dnn), ("SVR", y_pred_svr), ("KRR", y_pred_krr)]

for ax, (name, preds) in zip(axes, predictions):
    ax.scatter(y_test, preds, alpha=0.6, edgecolors='k')
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_title(f"{name}: Actual vs Predicted")
    ax.set_xlabel("Actual Voltage (V)")
    ax.set_ylabel("Predicted Voltage (V)")
    ax.grid(True)

plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png")
plt.close()
print("\nEvaluation complete! Saved metrics to 'results/' and models to 'models/'.")