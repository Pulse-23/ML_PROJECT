import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# =========================
# Custom Metrics Functions
# =========================
def wape(y_true, y_pred):
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency"""
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def vaf(y_true, y_pred):
    """Variance Accounted For"""
    return (1 - (np.var(y_true - y_pred) / np.var(y_true))) * 100

def lmi(y_true, y_pred):
    """Linear Model Index"""
    return 1 - (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true) + np.abs(y_pred)))

def rsr(y_true, y_pred):
    """RMSE-observations Standard deviation Ratio"""
    return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)

# =========================
# Load dataset
# =========================
df = pd.read_excel("D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx")

# Features and targets
X = df[['Cement(kg/m3)', 'Biomedical waste ash(kg/m3)', 
        'Fine aggregate(kg/m3)', 'Coarse aggregate(kg/m3)']]
y = df[['Compressive strength (28 days)(MPa)', 
        'Tensile strength(28 days)(MPa)', 
        'Flexural strength(28 days)(MPa)']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Random Forest + GridSearch
# =========================
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Predictions
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

y_train_pred_df = pd.DataFrame(y_train_pred, columns=y.columns, index=y_train.index)
y_test_pred_df = pd.DataFrame(y_test_pred, columns=y.columns, index=y_test.index)

# =========================
# Collect Metrics
# =========================
results = []
for col in y.columns:
    # Train
    train_metrics = {
        "Target": col,
        "Set": "Train",
        "R²": round(r2_score(y_train[col], y_train_pred_df[col]), 4),
        "WAPE (%)": round(wape(y_train[col], y_train_pred_df[col]), 2),
        "NSE": round(nse(y_train[col], y_train_pred_df[col]), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_train[col], y_train_pred_df[col])), 4),
        "VAF (%)": round(vaf(y_train[col], y_train_pred_df[col]), 2),
        "LMI": round(lmi(y_train[col], y_train_pred_df[col]), 4),
        "RSR": round(rsr(y_train[col], y_train_pred_df[col]), 4),
        "MAE": round(mean_absolute_error(y_train[col], y_train_pred_df[col]), 4)
    }
    # Test
    test_metrics = {
        "Target": col,
        "Set": "Test",
        "R²": round(r2_score(y_test[col], y_test_pred_df[col]), 4),
        "WAPE (%)": round(wape(y_test[col], y_test_pred_df[col]), 2),
        "NSE": round(nse(y_test[col], y_test_pred_df[col]), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test[col], y_test_pred_df[col])), 4),
        "VAF (%)": round(vaf(y_test[col], y_test_pred_df[col]), 2),
        "LMI": round(lmi(y_test[col], y_test_pred_df[col]), 4),
        "RSR": round(rsr(y_test[col], y_test_pred_df[col]), 4),
        "MAE": round(mean_absolute_error(y_test[col], y_test_pred_df[col]), 4)
    }
    results.extend([train_metrics, test_metrics])

    # Plot Actual vs Predicted (Test only)
    plt.figure(figsize=(6,4))
    plt.scatter(y_test[col], y_test_pred_df[col], alpha=0.7)
    plt.plot([y_test[col].min(), y_test[col].max()],
             [y_test[col].min(), y_test[col].max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted (Test) - {col}")
    plt.show()

# =========================
# Results DataFrame
# =========================
results_df = pd.DataFrame(results)
overall_avg = results_df.groupby("Set").mean(numeric_only=True).reset_index()
overall_avg["Target"] = "Overall Average"

# Reorder columns to match
overall_avg = overall_avg[results_df.columns]

# Append to results
final_results = pd.concat([results_df, overall_avg], ignore_index=True)

print("\n=== Final Random Forest Metrics with Overall Averages ===")
print(final_results)

print("\nBest Parameters Found:", grid_search.best_params_)

# Save trained model
joblib.dump(best_rf, "random_forest_model.pkl")
print("Model saved as random_forest_model.pkl")
