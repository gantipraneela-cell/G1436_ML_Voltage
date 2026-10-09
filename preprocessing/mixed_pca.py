import pandas as pd
from sklearn.decomposition import PCA
import joblib

print("=" * 60)
print("MIXED DATASET PCA")
print("=" * 60)

# Load scaled data
X_train_scaled = pd.read_csv(
    "mixed_X_train_scaled.csv"
)

X_test_scaled = pd.read_csv(
    "mixed_X_test_scaled.csv"
)

print("\nBefore PCA:")
print(
    "X_train_scaled:",
    X_train_scaled.shape
)

print(
    "X_test_scaled :",
    X_test_scaled.shape
)

# PCA
pca = PCA(n_components=80)

# Fit only on training data
X_train_pca = pca.fit_transform(
    X_train_scaled
)

# Transform test data
X_test_pca = pca.transform(
    X_test_scaled
)

# PCA column names
pca_columns = [
    f"PC{i+1}"
    for i in range(80)
]

# DataFrames
X_train_pca = pd.DataFrame(
    X_train_pca,
    columns=pca_columns
)

X_test_pca = pd.DataFrame(
    X_test_pca,
    columns=pca_columns
)

print("\nAfter PCA:")
print(
    "X_train_pca:",
    X_train_pca.shape
)

print(
    "X_test_pca :",
    X_test_pca.shape
)

# Explained variance
explained_variance = (
    pca.explained_variance_ratio_.sum()
)

print("\nTotal explained variance:")
print(
    f"{explained_variance * 100:.2f}%"
)

# Save PCA datasets
X_train_pca.to_csv(
    "mixed_X_train_pca.csv",
    index=False
)

X_test_pca.to_csv(
    "mixed_X_test_pca.csv",
    index=False
)

# Save variance information
variance_df = pd.DataFrame({
    "Principal_Component": pca_columns,
    "Explained_Variance_Ratio":
        pca.explained_variance_ratio_
})

variance_df.to_csv(
    "mixed_pca_explained_variance.csv",
    index=False
)

# Save PCA model
joblib.dump(
    pca,
    "mixed_pca.pkl"
)

print("\nFiles saved:")
print("mixed_X_train_pca.csv")
print("mixed_X_test_pca.csv")
print("mixed_pca_explained_variance.csv")
print("mixed_pca.pkl")

print("\n" + "=" * 60)
print("MIXED PCA COMPLETED")
print("=" * 60)