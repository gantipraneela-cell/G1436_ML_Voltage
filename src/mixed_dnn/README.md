# Mixed DNN - Experiment 2

## Overview
This experiment trains and evaluates a deep neural network regression model to predict average voltage from battery-material features.

## Data and preprocessing
- Dataset shape before preprocessing: 6,847 rows and 165 columns.
- Prediction target: `average_voltage`.
- Categorical feature: `working_ion`, one-hot encoded.
- Feature count after encoding: 173.
- Data split: 5,545 training rows, 617 validation rows, and 685 test rows.
- Regression-quantile stratification used 10 bins with `random_state=42`.
- `StandardScaler` and PCA were fitted on training data only.
- PCA retained 80 components, explaining 88.8811% of the variance.

## Model
The DNN architecture is 80 input features -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1, linear).

- Optimizer: Adam
- Loss: mean squared error (MSE)
- Metric: mean absolute error (MAE)
- Maximum epochs: 300
- Batch size: 32
- Early stopping: validation-loss patience of 30
- Learning-rate reduction: factor 0.5, patience 10, minimum learning rate 1e-7
- Recorded training: 145 epochs; best epoch 115

## Saved artifacts
- Model: `models/mixed_dnn/experiment2_DNN_model.keras`
- Preprocessed data: `data/preprocessed/mixed_dnn/`
- Evaluation metrics and predictions: `results/mixed_dnn/`
- Plots: `results/mixed_dnn/actual_vs_predicted.png`, `results/mixed_dnn/prediction_error_distribution.png`, and `results/mixed_dnn/error_vs_actual.png`

## Recorded test metrics
| Metric | Value |
|---|---:|
| MAE | 0.253700 |
| MSE | 0.248893 |
| RMSE | 0.498891 |
| R-squared (R2) | 0.921133 |

R2 is a regression metric and should not be described as classification accuracy.

## Scripts
- `train_experiment2_dnn.py`: trains and saves the DNN.
- `check_experiment2_features.py`: checks the feature data.
- `evaluate_experiment2.py`: evaluates the saved model.
- `error_by_voltage_range.py`: summarizes errors by voltage range.
- `plot_actual_vs_predicted.py`: plots actual versus predicted values.
- `plot_error_distribution.py`: plots the prediction-error distribution.
- `plot_error_vs_actual.py`: plots error against actual values.
- Preprocessing scripts are in `preprocessing/mixed_dnn/`.

## Reproducibility note
Run commands from the repository root so the relative paths resolve correctly. The training script is provided for reproducibility; rerunning it may replace the saved model in `models/mixed_dnn/`.
