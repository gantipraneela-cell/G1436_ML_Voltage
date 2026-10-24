import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("=" * 70)
print("DNN EXPERIMENT 2 - TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

# ------------------------------------------------------------
# 2. Load PCA data
# ------------------------------------------------------------

from pathlib import Path

DATA_DIR = Path("data/preprocessed/mixed_dnn")
MODEL_DIR = Path("models/mixed_dnn")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "E2_X_train_pca.csv")
X_val = pd.read_csv(DATA_DIR / "E2_X_val_pca.csv")

y_train = pd.read_csv(DATA_DIR / "E2_y_train.csv")
y_val = pd.read_csv(DATA_DIR / "E2_y_val.csv")
print("\nData:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)

# Convert to NumPy
X_train = X_train.astype(np.float32).values
X_val = X_val.astype(np.float32).values

y_train = y_train.astype(np.float32).values.ravel()
y_val = y_val.astype(np.float32).values.ravel()

# ------------------------------------------------------------
# 3. Build DNN
# ------------------------------------------------------------

model = Sequential([
    Input(shape=(80,)),

    Dense(128, activation="relu"),
    Dropout(0.20),

    Dense(64, activation="relu"),
    Dropout(0.20),

    Dense(32, activation="relu"),

    Dense(1, activation="linear")
])

# ------------------------------------------------------------
# 4. Compile
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("\nModel architecture:")
model.summary()

# ------------------------------------------------------------
# 5. Callbacks
# ------------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# ------------------------------------------------------------
# 6. Train
# ------------------------------------------------------------

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# ------------------------------------------------------------
# 7. Save model
# ------------------------------------------------------------

model.save(MODEL_DIR / "experiment2_DNN_model.keras")
# ------------------------------------------------------------
# 8. Training summary
# ------------------------------------------------------------

best_epoch = np.argmin(history.history["val_loss"]) + 1
best_val_loss = min(history.history["val_loss"])
best_val_mae = history.history["val_mae"][best_epoch - 1]

print("\n" + "=" * 70)
print("EXPERIMENT 2 TRAINING COMPLETED")
print("=" * 70)

print(f"\nTotal epochs trained : {len(history.history['loss'])}")
print(f"Best epoch          : {best_epoch}")
print(f"Best validation MSE : {best_val_loss:.6f}")
print(f"Best validation MAE : {best_val_mae:.6f}")

print("\nModel saved as:")
print(MODEL_DIR/"experiment2_DNN_model.keras")

print("\n" + "=" * 70)