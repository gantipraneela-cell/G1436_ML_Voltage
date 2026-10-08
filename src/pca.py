import pandas as pd
from sklearn.decomposition import PCA

# ============================================
# LOAD SCALED DATA
# ============================================

X_train_scaled = pd.read_csv("../X_train_scaled.csv")
X_test_scaled = pd.read_csv("../X_test_scaled.csv")

print("Scaled data shapes:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)


# ============================================
# PCA
# ============================================

# Keep 80 principal components
pca = PCA(n_components=80)

# IMPORTANT:
# Fit PCA ONLY on training data
X_train_pca = pca.fit_transform(X_train_scaled)

# Apply the same PCA transformation to test data
X_test_pca = pca.transform(X_test_scaled)


# ============================================
# CONVERT TO DATAFRAMES
# ============================================

pca_columns = [
    f"PC{i+1}"
    for i in range(80)
]

X_train_pca = pd.DataFrame(
    X_train_pca,
    columns=pca_columns
)

X_test_pca = pd.DataFrame(
    X_test_pca,
    columns=pca_columns
)


# ============================================
# EXPLAINED VARIANCE
# ============================================

explained_variance = pca.explained_variance_ratio_

cumulative_variance = explained_variance.cumsum()

print("\n====================================")
print("PCA RESULTS")
print("====================================")

print("Original number of features:", X_train_scaled.shape[1])
print("Number of PCA components   :", X_train_pca.shape[1])

print(
    "\nVariance explained by 80 components:",
    cumulative_variance[-1]
)

print(
    "Percentage of variance explained:",
    cumulative_variance[-1] * 100,
    "%"
)


# ============================================
# SHOW VARIANCE AT DIFFERENT COMPONENT COUNTS
# ============================================

print("\nCumulative explained variance:")

for n in [10, 20, 30, 40, 50, 60, 70, 80]:
    print(
        f"{n} components: "
        f"{cumulative_variance[n-1] * 100:.2f}%"
    )


# ============================================
# SAVE PCA DATA
# ============================================

X_train_pca.to_csv("X_train_pca.csv", index=False)
X_test_pca.to_csv("X_test_pca.csv", index=False)

# Save explained variance information

variance_df = pd.DataFrame({
    "Component": range(1, 81),
    "Explained_Variance_Ratio": explained_variance,
    "Cumulative_Explained_Variance": cumulative_variance
})

variance_df.to_csv(
    "../pca_explained_variance.csv",
    index=False
)


print("\nFiles saved successfully.")
print("X_train_pca.csv")
print("X_test_pca.csv")
print("pca_explained_variance.csv")