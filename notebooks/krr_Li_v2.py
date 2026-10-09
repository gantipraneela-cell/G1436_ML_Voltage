import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading enriched Materials Project v2 dataset for KRR...")
    X_train = pd.read_csv('preprocessing/X_train_li_v2.csv')
    X_test  = pd.read_csv('preprocessing/X_test_li_v2.csv')
    y_train = pd.read_csv('preprocessing/y_train_li_v2.csv').values.ravel()
    y_test  = pd.read_csv('preprocessing/y_test_li_v2.csv').values.ravel()

    # Kernel Ridge Regression with RBF Kernel
    print("Training Kernel Ridge Regression (KRR) Model...")
    krr = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.01)
    krr.fit(X_train, y_train)

    y_pred = krr.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = np.mean(np.abs(y_test - y_pred) <= 0.5) * 100

    print("\n==========================================")
    print("   KERNEL RIDGE REGRESSION RESULTS (V2)")
    print("==========================================")
    print(f"Mean Absolute Error (MAE) : {mae:.4f} V")
    print(f"Root Mean Sq. Error (RMSE): {rmse:.4f} V")
    print(f"R2 Score                   : {r2:.4f} ({r2*100:.2f}%)")
    print(f"Mean Abs. Pct. Error(MAPE): {mape:.2f}%")
    print(f"Accuracy (within ±0.5 V)  : {accuracy:.2f}%")
    print("==========================================\n")

if __name__ == '__main__':
    main()