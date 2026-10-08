import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================
# LOAD TRAINING AND TEST DATA
# ============================================

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

print("Original shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================
# FEATURE SCALING
# ============================================

scaler = StandardScaler()

# IMPORTANT:
# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler for test data
X_test_scaled = scaler.transform(X_test)


# ============================================
# CONVERT TO DATAFRAME
# ============================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)


# ============================================
# CHECK RESULTS
# ============================================

print("\n====================================")
print("FEATURE SCALING RESULTS")
print("====================================")

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape :", X_test_scaled.shape)

print("\nFirst 5 rows of scaled training data:")
print(X_train_scaled.head())

print("\nMean of first 5 scaled features:")
print(X_train_scaled.iloc[:, :5].mean())

print("\nStandard deviation of first 5 scaled features:")
print(X_train_scaled.iloc[:, :5].std())


# ============================================
# SAVE SCALED DATA
# ============================================

X_train_scaled.to_csv("../X_train_scaled.csv", index=False)
X_test_scaled.to_csv("../X_test_scaled.csv", index=False)

print("\nFiles saved successfully.")