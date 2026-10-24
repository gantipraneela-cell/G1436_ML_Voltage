\# Battery Average Voltage Prediction using SVR



\## Project

This project evaluates Support Vector Regression (SVR) for predicting

average battery voltage from material features.



\## Recorded comparison

The existing comparison file reports these results:



| Dataset | Input features | MAE (V) | RMSE (V) | R² |

|---|---:|---:|---:|---:|

| Original | 164 | 0.2130 | 0.5893 | 0.9017 |

| Cleaned | 119 | 0.6085 | 1.2193 | 0.5793 |



\## Important limitation

The 90.17% R² result is from the original feature set. Potential

feature leakage and extreme voltage records are still under review.

It must not be interpreted as a confirmed leakage-free result.



The Round 4 tuning experiment is separate and does not reproduce

the 90.17% test result.



\## Files

\- `src/compare\_svr\_original\_vs\_clean.py`: original-versus-cleaned comparison

\- `src/tune\_SVR\_60features\_round4.py`: separate Round 4 tuning experiment

\- `data/`: original and cleaned feature datasets

\- `results/`: recorded metrics and Round 4 validation results



\## Reproducibility

Run scripts only after checking their expected input paths and outputs.

The comparison script reads the CSV files from the current working

directory and writes its comparison CSV there.

