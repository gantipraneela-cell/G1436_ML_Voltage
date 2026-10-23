import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/preprocessed/mixed_dnn")
from sklearn.decomposition import PCA

print("=" * 70)
print("EXPERIMENT 2 - STANDARDIZATION + PCA")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load train / validation / test data
# ------------------------------------------------------------
X_train = pd.read_csv(DATA_DIR / "E2_X_train_raw.csv")
X_val = pd.read_csv(DATA_DIR / "E2_X_val_raw.csv")
X_test = pd.read_csv(DATA_DIR / "E2_X_test_raw.csv")
print("\nInput shapes:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

# ------------------------------------------------------------
# 2. Convert to float
# ------------------------------------------------------------

X_train = X_train.astype(np.float64)
X_val = X_val.astype(np.float64)
X_test = X_test.astype(np.float64)

# ------------------------------------------------------------
# 3. StandardScaler
#    IMPORTANT: fit ONLY on training data
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nStandardization completed.")

# ------------------------------------------------------------
# 4. PCA
#    Fit ONLY on training data
# ------------------------------------------------------------

pca = PCA(n_components=80)

X_train_pca = pca.fit_transform(X_train_scaled)

X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_variance = pca.explained_variance_ratio_.sum()

print("\nPCA completed.")
print("Original features :", X_train.shape[1])
print("PCA components    :", X_train_pca.shape[1])
print(
    "Explained variance:",
    f"{explained_variance * 100:.4f}%"
)

# ------------------------------------------------------------
# 5. Save PCA datasets
# ------------------------------------------------------------

train_columns = [
    f"PC{i}"
    for i in range(1, 81)
]

X_train_pca_df = pd.DataFrame(
    X_train_pca,
    columns=train_columns
)

X_val_pca_df = pd.DataFrame(
    X_val_pca,
    columns=train_columns
)

X_test_pca_df = pd.DataFrame(
    X_test_pca,
    columns=train_columns
)

X_train_pca_df.to_csv(DATA_DIR / "E2_X_train_pca.csv", index=False)

X_val_pca_df.to_csv(DATA_DIR / "E2_X_val_pca.csv", index=False)

X_test_pca_df.to_csv(DATA_DIR / "E2_X_test_pca.csv", index=False)

print("\nFiles saved:")
print("E2_X_train_pca.csv")
print("E2_X_val_pca.csv")
print("E2_X_test_pca.csv")

print("\n" + "=" * 70)
print("EXPERIMENT 2 PCA COMPLETED")
print("=" * 70)