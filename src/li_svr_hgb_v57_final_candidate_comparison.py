
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# V57: FINAL CANDIDATE COMPARISON
# Four fixed candidates, identical grouped CV splits.
# Held-out test data is NOT loaded.
# ============================================================

DATA_PATH = "data/preprocessed/li_v2_train_composition.csv"
SUMMARY_PATH = "data/preprocessed/li_svr_hgb_v57_summary.csv"
OOF_PATH = "data/preprocessed/li_svr_hgb_v57_oof.csv"

N_SPLITS = 5
SVR_WEIGHT = 0.60
HGB_WEIGHT = 0.40

CANDIDATES = [
    {
        "name": "Baseline_C50_Gamma005_Eps025",
        "C": 50,
        "gamma": 0.005,
        "epsilon": 0.25,
    },
    {
        "name": "Candidate_C100_Gamma005_Eps025",
        "C": 100,
        "gamma": 0.005,
        "epsilon": 0.25,
    },
    {
        "name": "Candidate_C75_Gamma005_Eps025",
        "C": 75,
        "gamma": 0.005,
        "epsilon": 0.25,
    },
    {
        "name": "Candidate_C50_Gamma007_Eps025",
        "C": 50,
        "gamma": 0.007,
        "epsilon": 0.25,
    },
]


def calculate_metrics(actual, predicted):
    error = predicted - actual
    absolute_error = np.abs(error)

    return {
        "within_0.5V_percent": 100 * np.mean(absolute_error <= 0.5),
        "within_1.0V_percent": 100 * np.mean(absolute_error <= 1.0),
        "within_1.5V_percent": 100 * np.mean(absolute_error <= 1.5),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "bias": np.mean(error),
    }


def main():
    print("=" * 72)
    print("V57: FINAL CANDIDATE COMPARISON")
    print("=" * 72)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    target_col = "average_voltage"
    formula_col = "battery_formula"

    for col in [target_col, formula_col]:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    y = df[target_col].to_numpy(dtype=float)
    groups = df[formula_col].fillna("").astype(str).to_numpy()

    X_df = df.drop(columns=[target_col, formula_col])
    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.dropna(axis=1, how="all")
    X_df = X_df.loc[:, X_df.nunique(dropna=True) > 1]
    X = X_df.to_numpy(dtype=float)

    if not np.isfinite(y).all():
        raise ValueError("Target contains missing or infinite values.")

    if len(np.unique(groups)) < N_SPLITS:
        raise ValueError("Not enough unique formulas for GroupKFold.")

    print(f"Rows: {len(df)}")
    print(f"Numeric features: {X.shape[1]}")
    print(f"Unique formulas: {len(np.unique(groups))}")
    print(f"Candidates: {len(CANDIDATES)}")
    print("Blend: 60% SVR + 40% HGB")

    # Build the splits once so every candidate uses identical folds.
    cv = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(X, y, groups))

    results = []
    oof_outputs = []

    for candidate_number, candidate in enumerate(CANDIDATES, start=1):
        print(
            f"\nCandidate {candidate_number}/{len(CANDIDATES)}: "
            f"{candidate['name']}"
        )

        svr_oof = np.full(len(y), np.nan)
        hgb_oof = np.full(len(y), np.nan)

        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train = y[train_idx]

            svr = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                SVR(
                    kernel="rbf",
                    C=candidate["C"],
                    gamma=candidate["gamma"],
                    epsilon=candidate["epsilon"],
                ),
            )

            hgb = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            )

            svr.fit(X_train, y_train)
            hgb.fit(X_train, y_train)

            svr_oof[valid_idx] = svr.predict(X_valid)
            hgb_oof[valid_idx] = hgb.predict(X_valid)

            print(f"  Fold {fold}/{N_SPLITS} completed")

        if np.isnan(svr_oof).any() or np.isnan(hgb_oof).any():
            raise RuntimeError("Some OOF predictions were not generated.")

        hybrid_oof = (
            SVR_WEIGHT * svr_oof + HGB_WEIGHT * hgb_oof
        )

        result = {
            "candidate": candidate["name"],
            "C": candidate["C"],
            "gamma": candidate["gamma"],
            "epsilon": candidate["epsilon"],
            **calculate_metrics(y, hybrid_oof),
        }
        results.append(result)

        candidate_output = pd.DataFrame({
            "battery_formula": groups,
            "actual_voltage": y,
            "candidate": candidate["name"],
            "svr_prediction": svr_oof,
            "hgb_prediction": hgb_oof,
            "hybrid_prediction": hybrid_oof,
        })
        oof_outputs.append(candidate_output)

        print(
            f"  +/-0.5 V: {result['within_0.5V_percent']:.2f}% | "
            f"+/-1.0 V: {result['within_1.0V_percent']:.2f}% | "
            f"MAE: {result['MAE']:.4f} V"
        )

    summary = pd.DataFrame(results).sort_values(
        ["within_0.5V_percent", "MAE"],
        ascending=[False, True],
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    pd.concat(oof_outputs, ignore_index=True).to_csv(
        OOF_PATH, index=False
    )

    print("\n" + "=" * 72)
    print("V57 FINAL RANKING")
    print("=" * 72)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nFILES SAVED")
    print(SUMMARY_PATH)
    print(OOF_PATH)

    print("\nInterpretation:")
    print("- Rank primarily by within_0.5V_percent.")
    print("- Use MAE and RMSE to compare ties and trade-offs.")
    print("- These are grouped CV results, not held-out test results.")
    print("- The held-out test benchmark remains unchanged.")


if __name__ == "__main__":
    main()
