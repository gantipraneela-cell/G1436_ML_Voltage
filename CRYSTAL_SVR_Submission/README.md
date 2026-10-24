# Battery Average Voltage Prediction Using SVR

## Project
This submission evaluates Support Vector Regression (SVR) for predicting average battery voltage from material features.

## Recorded results
The existing comparison file records the following test metrics:

| Dataset | Input features | MAE (V) | RMSE (V) | R-squared |
|---|---:|---:|---:|---:|
| Original | 164 | 0.2130 | 0.5893 | 0.9017 |
| Cleaned | 119 | 0.6085 | 1.2193 | 0.5793 |

The original result corresponds to approximately 90.17% R-squared. Potential feature leakage and extreme target values remain under review; this result must not be described as confirmed leakage-free performance.

The separate Round 4 tuning results file has a highest listed R-squared of approximately 0.6243. It is a separate experiment and does not reproduce the original 90.17% result.

## Extreme-voltage audit
The `audit/` directory contains:
- `extreme_voltage_audit_summary.txt`: summary of numerical source-energy consistency checks for four highlighted records.
- `extreme_voltage_target_audit.csv`: 24 candidate extreme-voltage records with source-voltage values, material IDs, and candidate row indices.

**Audit limitation:** The audit checks numerical consistency with the inspected or embedded source-energy entries. It does not establish that every voltage is physically valid, that the source entries are the correct thermodynamic states, or that the model is reliable on extreme values. Further source and thermodynamic verification is required.

## Files
- `src/compare_svr_original_vs_clean.py`: original-versus-cleaned comparison script.
- `src/tune_SVR_60features_round4.py`: separate Round 4 tuning experiment.
- `data/`: original and cleaned feature datasets.
- `results/`: recorded comparison metrics and tuning results.
- `audit/`: extreme-voltage audit report and target-record CSV.

## Reproducibility and status
The reported metrics are the recorded results in the included CSV files; this README does not claim that the experiments have been rerun during submission preparation. The audit files do not modify the datasets or retrain SVR. Check the scripts' expected input paths before running them.