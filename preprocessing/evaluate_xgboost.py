import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading enriched Materials Project v2 dataset...")
    # 1. Load Enriched Li-v2 Dataset
    X_train_df = pd.read_csv('preprocessing/X_train_li_v2.csv')
    X_test_df  = pd.read_csv('preprocessing/X_test_li_v2.csv')
    y_train    = pd.read_csv('preprocessing/y_train_li_v2.csv').values.ravel()
    y_test     = pd.read_csv('preprocessing/y_test_li_v2.csv').values.ravel()

    print(f"Loaded {X_train_df.shape[1]} features for training.")

    # 2. Target Transformation (QuantileTransformer for non-gaussian target distribution)
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    y_train_transformed = qt.fit_transform(y_train.reshape(-1, 1)).ravel()

    # 3. Base Estimators for Stacking Ensemble
    base_models = [
        ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)),
        ('lgbm', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=-1)),
        ('cat', CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_seed=42)),
        ('et', ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))
    ]

    # Meta-Learner over base predictions
    meta_model = XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42)

    stacking_ensemble = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    # 4. Train Model
    print("\nTraining Peak Stacking Ensemble on 137 v2 features...")
    stacking_ensemble.fit(X_train_df, y_train_transformed)

    # 5. Predict & Inverse Transform Targets
    y_pred_transformed = stacking_ensemble.predict(X_test_df)
    y_pred = qt.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()

    # 6. Metric Calculations
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = np.mean(np.abs(y_test - y_pred) <= 0.5) * 100

    # 7. Print Results
    print("\n==========================================")
    print("  PEAK TREE STACKING RESULTS (LI-ONLY V2)")
    print("==========================================")
    print(f"Mean Absolute Error (MAE) : {mae:.4f} V")
    print(f"Root Mean Sq. Error (RMSE): {rmse:.4f} V")
    print(f"R2 Score                   : {r2:.4f} ({r2*100:.2f}%)")
    print(f"Mean Abs. Pct. Error(MAPE): {mape:.2f}%")
    print(f"Accuracy (within ±0.5 V)  : {accuracy:.2f}%")
    print("==========================================\n")

if __name__ == '__main__':
    main()