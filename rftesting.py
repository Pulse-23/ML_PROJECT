import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ========================
# Load trained model
# ========================
model_path = "random_forest_model.pkl"   # Must match the file used during training
best_rf = joblib.load(model_path)

# ========================
# Load dataset
# ========================
df = pd.read_excel("D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx")

# Features and targets
X = df[['Cement(kg/m3)', 'Biomedical waste ash(kg/m3)', 
        'Fine aggregate(kg/m3)', 'Coarse aggregate(kg/m3)']]
y = df[['Compressive strength (28 days)(MPa)', 
        'Tensile strength(28 days)(MPa)', 
        'Flexural strength(28 days)(MPa)']]

# Split dataset (must be same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# Predictions
# ========================
y_pred = best_rf.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns, index=y_test.index)

# ========================
# Evaluation
# ========================
results = []

for col in y.columns:
    mse = mean_squared_error(y_test[col], y_pred_df[col])
    r2 = r2_score(y_test[col], y_pred_df[col])
    mae = mean_absolute_error(y_test[col], y_pred_df[col])
    accuracy = 100 * (1 - (mae / y_test[col].mean()))  # regression "accuracy"

    results.append({
        "Target": col,
        "MSE": round(mse, 4),
        "R²": round(r2, 4),
        "MAE": round(mae, 4),
        "Accuracy (%)": round(accuracy, 2)
    })

    # Plot Actual vs Predicted
    plt.figure(figsize=(6,4))
    plt.scatter(y_test[col], y_pred_df[col], alpha=0.7)
    plt.plot([y_test[col].min(), y_test[col].max()],
             [y_test[col].min(), y_test[col].max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted - {col}")
    plt.show()

# ========================
# Print results
# ========================
results_df = pd.DataFrame(results)
print("\n=== Random Forest Testing Results ===")
print(results_df)
