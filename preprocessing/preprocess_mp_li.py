import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matminer.featurizers.composition import ElementProperty
import warnings

warnings.filterwarnings('ignore')

def main():
    print("1. Loading raw Materials Project dataset...")
    df = pd.read_csv('preprocessing/raw_mp_li_v2.csv')

    # Drop rows missing average_voltage or discharge formula
    df = df.dropna(subset=['average_voltage', 'formula_discharge']).reset_index(drop=True)

    print("2. Extracting Magpie physical descriptors (single-process mode)...")
    # n_jobs=1 prevents Windows multiprocessing loops
    featurizer = ElementProperty.from_preset(preset_name="magpie")
    featurizer.set_n_jobs(1)
    
    df_features = featurizer.featurize_dataframe(df, col_id="formula_discharge", ignore_errors=True)

    # Select numeric elemental descriptors + MP structural/property columns
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    target_col = 'average_voltage'
    exclude_cols = [target_col, 'battery_id', 'num_steps']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    X = df_features[feature_cols].fillna(0)
    y = df_features[target_col].values

    print(f"Dataset Feature Matrix Shape: {X.shape}")

    # 3. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 5. Save v2 files safely inside preprocessing/
    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv('preprocessing/X_train_li_v2.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv('preprocessing/X_test_li_v2.csv', index=False)
    pd.DataFrame(y_train, columns=['average_voltage']).to_csv('preprocessing/y_train_li_v2.csv', index=False)
    pd.DataFrame(y_test, columns=['average_voltage']).to_csv('preprocessing/y_test_li_v2.csv', index=False)

    print("\nSuccess! Processed v2 files saved:")
    print(" - preprocessing/X_train_li_v2.csv")
    print(" - preprocessing/X_test_li_v2.csv")
    print(" - preprocessing/y_train_li_v2.csv")
    print(" - preprocessing/y_test_li_v2.csv")

if __name__ == '__main__':
    main()