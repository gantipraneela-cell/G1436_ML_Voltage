import numpy as np

# 1. Predict on the test set using your trained DNN model
y_pred_dnn = model.predict(X_test).squeeze()

# 2. Calculate absolute errors against true test values
absolute_errors = np.abs(y_test - y_pred_dnn)

# 3. Compute accuracy percentage within the ±0.5V threshold
tolerance = 0.5
accuracy_within_tolerance = np.mean(absolute_errors <= tolerance) * 100

# 4. Print the result so it shows up in your terminal
print(f"DNN Accuracy (within ±{tolerance} V): {accuracy_within_tolerance:.2f}%")