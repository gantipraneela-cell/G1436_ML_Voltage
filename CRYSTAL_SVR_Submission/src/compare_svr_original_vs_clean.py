import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "ORIGINAL": os.path.join(
        BASE_DIR, "data", "battery_mixed_features.csv"
    ),
    "CLEANED": os.path.join(
        BASE_DIR, "data", "battery_mixed_clean_features.csv"
    )
}

RESULT_FILE = os.path.join(
    BASE_DIR, "results", "svr_original_vs_clean_comparison.csv"
)

results = []

for name, filename in FILES.items():
    print("\n" + "=" * 55)
    print("DATASET:", name)
    print("=" * 55)

    data = pd.read_csv(filename)

    required = ["average_voltage", "working_ion"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns in {filename}: {missing}")

    X = data.drop(columns=["average_voltage"])
    y = data["average_voltage"]

    cat_cols = ["working_ion"]
    num_cols = [col for col in X.columns if col not in cat_cols]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectKBest(score_func=f_regression, k=60)),
        ("svr", SVR(
            kernel="rbf",
            C=180,
            gamma=0.004,
            epsilon=0.08
        ))
    ])

    model.fit(X_dev, y_dev)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Rows:", len(data))
    print("Input features:", X.shape[1])
    print("Test MAE:", round(mae, 6))
    print("Test RMSE:", round(rmse, 6))
    print("Test R2:", round(r2, 6))
    print("Test R2 percentage:", round(r2 * 100, 2), "%")

    results.append({
        "Dataset": name,
        "Input_Features": X.shape[1],
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2,
        "Test_R2_Percent": r2 * 100
    })

summary = pd.DataFrame(results)

os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
summary.to_csv(RESULT_FILE, index=False)

print("\nFINAL COMPARISON")
print(summary.to_string(index=False))
print("\nSaved:", RESULT_FILE)