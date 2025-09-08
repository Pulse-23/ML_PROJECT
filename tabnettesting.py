import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset (for testing)
df = pd.read_excel("D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx")

# Features and targets
X = df[['Cement(kg/m3)', 'Biomedical waste ash(kg/m3)', 'Fine aggregate(kg/m3)', 'Coarse aggregate(kg/m3)']]
y = df[['Compressive strength (28 days)(MPa)', 
        'Tensile strength(28 days)(MPa)', 
        'Flexural strength(28 days)(MPa)']]

# Train-test split (use same seed as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model = joblib.load("biomedical_rf_model.pkl")
print("✅ Model loaded successfully!")

# Predictions
y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns, index=y_test.index)

# Evaluation metrics
accuracies = []
for col in y.columns:
    mse = mean_squared_error(y_test[col], y_pred_df[col])
    r2 = r2_score(y_test[col], y_pred_df[col])
    mae = mean_absolute_error(y_test[col], y_pred_df[col])
    accuracy = 100 * (1 - (mae / y_test[col].mean()))  # regression "accuracy"
    accuracies.append(accuracy)

    print(f"\n🔹 Results for {col}:")
    print("MSE:", round(mse, 4))
    print("R2 Score:", round(r2, 4))
    print("MAE:", round(mae, 4))
    print("Accuracy:", round(accuracy, 2), "%")

# Overall average accuracy
overall_accuracy = np.mean(accuracies)
print("\n✅ Overall Accuracy across all outputs:", round(overall_accuracy, 2), "%")
