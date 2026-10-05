import pandas as pd
from sklearn.decomposition import PCA

print("=" * 60)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 60)

# Load scaled data
X_train_scaled = pd.read_csv("X_train_scaled.csv")
X_test_scaled = pd.read_csv("X_test_scaled.csv")

print("\nBefore PCA:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)

# ---------------------------------------------------------
# Step 1: Fit PCA on TRAINING data only
# ---------------------------------------------------------

pca_full = PCA()
pca_full.fit(X_train_scaled)

# Cumulative explained variance
cumulative_variance = pca_full.explained_variance_ratio_.cumsum()

print("\nExplained variance:")

for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 117]:
    print(
        f"{n:3d} components -> "
        f"{cumulative_variance[n-1] * 100:.2f}% variance"
    )

# ---------------------------------------------------------
# Step 2: Choose number of components
# ---------------------------------------------------------

n_components = 80

print("\nSelected components:", n_components)
print(
    f"Variance retained: "
    f"{cumulative_variance[n_components-1] * 100:.2f}%"
)

# ---------------------------------------------------------
# Step 3: Fit PCA with selected components
# ---------------------------------------------------------

pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train_scaled)

# IMPORTANT:
# Test data is only transformed, NOT fitted
X_test_pca = pca.transform(X_test_scaled)

print("\nAfter PCA:")
print("X_train_pca:", X_train_pca.shape)
print("X_test_pca :", X_test_pca.shape)

# ---------------------------------------------------------
# Step 4: Save PCA datasets
# ---------------------------------------------------------

train_pca_df = pd.DataFrame(
    X_train_pca,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

test_pca_df = pd.DataFrame(
    X_test_pca,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

train_pca_df.to_csv("X_train_pca.csv", index=False)
test_pca_df.to_csv("X_test_pca.csv", index=False)

# Save explained variance
variance_df = pd.DataFrame({
    "Component": range(1, len(pca.explained_variance_ratio_) + 1),
    "Explained_Variance": pca.explained_variance_ratio_,
    "Cumulative_Variance": pca.explained_variance_ratio_.cumsum()
})

variance_df.to_csv("pca_explained_variance.csv", index=False)

print("\nFiles created:")
print("X_train_pca.csv")
print("X_test_pca.csv")
print("pca_explained_variance.csv")

print("\nPCA completed successfully!")