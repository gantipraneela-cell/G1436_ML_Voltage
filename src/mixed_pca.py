import pandas as pd
from sklearn.decomposition import PCA

print("=" * 60)
print("MIXED DATA PCA")
print("=" * 60)

# Load scaled Mixed data
X_train_scaled = pd.read_csv("mixed_X_train_scaled.csv")
X_test_scaled = pd.read_csv("mixed_X_test_scaled.csv")

print("\nBefore PCA:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)

# --------------------------------------------------
# Find explained variance using all components
# --------------------------------------------------

pca_full = PCA()
pca_full.fit(X_train_scaled)

cumulative_variance = pca_full.explained_variance_ratio_.cumsum()

print("\nExplained variance:")

check_points = [
    10, 20, 30, 40, 50, 60,
    70, 80, 90, 100
]

for n in check_points:
    if n <= X_train_scaled.shape[1]:
        print(
            f"{n:3d} components -> "
            f"{cumulative_variance[n-1] * 100:.2f}% variance"
        )

# Also show 95% and 97% variance component counts
n_95 = next(
    i + 1 for i, v in enumerate(cumulative_variance)
    if v >= 0.95
)

n_97 = next(
    i + 1 for i, v in enumerate(cumulative_variance)
    if v >= 0.97
)

print("\nComponents needed for 95% variance:", n_95)
print("Components needed for 97% variance:", n_97)

# --------------------------------------------------
# Use 80 components for comparison with Li workflow
# --------------------------------------------------

n_components = min(80, X_train_scaled.shape[1])

print("\nSelected components:", n_components)
print(
    f"Variance retained: "
    f"{cumulative_variance[n_components-1] * 100:.2f}%"
)

# --------------------------------------------------
# Fit PCA ONLY on training data
# --------------------------------------------------

pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train_scaled)

# Transform test data using the same PCA
X_test_pca = pca.transform(X_test_scaled)

print("\nAfter PCA:")
print("X_train_pca:", X_train_pca.shape)
print("X_test_pca :", X_test_pca.shape)

# --------------------------------------------------
# Save PCA datasets
# --------------------------------------------------

train_pca_df = pd.DataFrame(
    X_train_pca,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

test_pca_df = pd.DataFrame(
    X_test_pca,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

train_pca_df.to_csv("mixed_X_train_pca.csv", index=False)
test_pca_df.to_csv("mixed_X_test_pca.csv", index=False)

# Save explained variance
variance_df = pd.DataFrame({
    "Component": range(
        1,
        len(pca.explained_variance_ratio_) + 1
    ),
    "Explained_Variance":
        pca.explained_variance_ratio_,
    "Cumulative_Variance":
        pca.explained_variance_ratio_.cumsum()
})

variance_df.to_csv(
    "mixed_pca_explained_variance.csv",
    index=False
)

print("\nFiles created:")
print("mixed_X_train_pca.csv")
print("mixed_X_test_pca.csv")
print("mixed_pca_explained_variance.csv")

print("\nMixed PCA completed successfully!")