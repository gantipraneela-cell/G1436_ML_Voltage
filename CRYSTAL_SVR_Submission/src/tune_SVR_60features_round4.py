import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use the submission folder as the working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR, "data", "battery_mixed_features.csv"
)
RESULT_FILE = os.path.join(
    BASE_DIR, "results", "svr_60features_round4_results.csv"
)

# Load and validate data
data = pd.read_csv(DATA_FILE)

required_columns = ["average_voltage", "working_ion",
                    "energy_grav", "energy_vol"]

missing_columns = [
    col for col in required_columns if col not in data.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if len(data) < 10:
    raise ValueError("Dataset has too few rows.")

X = data.drop(columns=["average_voltage", "energy_grav", "energy_vol"])
y = data["average_voltage"]

cat_cols = ["working_ion"]
num_cols = [c for c in X.columns if c not in cat_cols]

if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Dataset contains missing values. Check data before training.")

if X.shape[1] < 60:
    raise ValueError(
        f"Only {X.shape[1]} input features available; at least 60 are required."
    )

# Split data into development and test sets
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Split development data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.20, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Original Round 4 parameter grid
param_grid = {
    "C": [170, 180, 190, 200, 220],
    "gamma": [0.00375, 0.004, 0.00425],
    "epsilon": [0.07, 0.08, 0.09]
}

results = []

print("Starting Round 4: 45 combinations")
print("Dataset rows:", len(data))
print("Input features before selection:", X.shape[1])
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# Tune parameters using validation data only
for i, p in enumerate(ParameterGrid(param_grid), 1):
    model = Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectKBest(score_func=f_regression, k=60)),
        ("svr", SVR(
            kernel="rbf",
            C=p["C"],
            gamma=p["gamma"],
            epsilon=p["epsilon"]
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    results.append({
        "C": p["C"],
        "gamma": p["gamma"],
        "epsilon": p["epsilon"],
        "MAE": mean_absolute_error(y_val, pred),
        "RMSE": np.sqrt(mean_squared_error(y_val, pred)),
        "R2": r2_score(y_val, pred)
    })

    print(
        f"[{i}/45] C={p['C']} gamma={p['gamma']} "
        f"epsilon={p['epsilon']} R2={results[-1]['R2']:.6f}"
    )

# Sort by validation R2
results_df = pd.DataFrame(results).sort_values(
    "R2", ascending=False
)

print("\nTOP 10 VALIDATION RESULTS")
print(results_df.head(10).to_string(index=False))

# Save tuning results
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
results_df.to_csv(RESULT_FILE, index=False)
print("\nSaved:", RESULT_FILE)

# Select best validation parameters
best = results_df.iloc[0]
best_C = best["C"]
best_gamma = best["gamma"]
best_epsilon = best["epsilon"]

print("\nBest validation parameters:")
print("C:", best_C)
print("gamma:", best_gamma)
print("epsilon:", best_epsilon)
print("Validation R2:", best["R2"])

# Train final model on the full development set
final_model = Pipeline([
    ("preprocess", preprocessor),
    ("select", SelectKBest(score_func=f_regression, k=60)),
    ("svr", SVR(
        kernel="rbf",
        C=best_C,
        gamma=best_gamma,
        epsilon=best_epsilon
    ))
])

final_model.fit(X_dev, y_dev)
test_pred = final_model.predict(X_test)

# Evaluate once on the held-out test set
test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print("\nFINAL TEST RESULTS")
print("Selected features:", 60)
print("C:", best_C)
print("gamma:", best_gamma)
print("epsilon:", best_epsilon)
print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)
print("Test R2:", test_r2)
print("Test R2 percentage:", round(test_r2 * 100, 2), "%")

print(
    "\nReference only: the previously recorded 0.9017408030 "
    "R2 came from a different experiment and is not the result of this run."
)