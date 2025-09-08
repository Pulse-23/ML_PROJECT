import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# ================= Load dataset =================
df = pd.read_excel("D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx")

# Features and targets
X = df[['Cement(kg/m3)', 'Biomedical waste ash(kg/m3)', 'Fine aggregate(kg/m3)', 'Coarse aggregate(kg/m3)']]
y = df[['Compressive strength (28 days)(MPa)', 
        'Tensile strength(28 days)(MPa)', 
        'Flexural strength(28 days)(MPa)']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= Model Training =================
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

# Save model
joblib.dump(best_rf, "biomedical_rf_model.pkl")
print("✅ Model saved as biomedical_rf_model.pkl")

# ================= Custom Regression Metrics =================
def wape(y_true, y_pred):
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def nash_sutcliffe(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def vaf(y_true, y_pred):
    return (1 - (np.var(y_true - y_pred) / np.var(y_true))) * 100

def rsr(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)

def lin_concordance_corr(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)

# ================= Evaluate Function =================
def evaluate_all_metrics(y_true, y_pred, dataset_name):
    results = []
    for i, col in enumerate(y.columns):
        yt, yp = y_true.iloc[:, i], y_pred[:, i]

        metrics = {
            "Dataset": dataset_name,
            "Target": col,
            "R2": round(r2_score(yt, yp), 4),
            "WAPE": round(wape(yt, yp), 4),
            "NS": round(nash_sutcliffe(yt, yp), 4),
            "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 4),
            "VAF": round(vaf(yt, yp), 4),
            "LMI": round(lin_concordance_corr(yt, yp), 4),
            "RSR": round(rsr(yt, yp), 4),
            "MAE": round(mean_absolute_error(yt, yp), 4),
        }
        results.append(metrics)
    return results

# ================= Results Table =================
train_results = evaluate_all_metrics(y_train, y_train_pred, "Train")
test_results = evaluate_all_metrics(y_test, y_test_pred, "Test")

results_df = pd.DataFrame(train_results + test_results)

# === Compute average row per dataset ===
avg_df = (
    results_df
    .groupby("Dataset")
    .mean(numeric_only=True)   # take mean of numeric metrics
    .reset_index()
)

avg_df["Target"] = "Overall Average"  # mark as overall row

# Reorder columns to match results_df
avg_df = avg_df[results_df.columns]

# Combine detailed + average rows
final_df = pd.concat([results_df, avg_df], ignore_index=True)

print("\n=== Final Results Table (with overall averages) ===")
print(final_df)

# Save results
final_df.to_excel("model_results_with_avg.xlsx", index=False)
print("✅ Detailed results with averages saved as model_results_with_avg.xlsx")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

def plot_tabnet_3d(model, X, y, feature_names, f1_idx=0, f2_idx=1, target_idx=0):
    """
    target_idx: which output to visualize (0=compressive, 1=tensile, 2=flexural)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Take two features
    f1 = X.iloc[:, f1_idx]
    f2 = X.iloc[:, f2_idx]

    # Create meshgrid
    f1_range = np.linspace(f1.min(), f1.max(), 40)
    f2_range = np.linspace(f2.min(), f2.max(), 40)
    F1, F2 = np.meshgrid(f1_range, f2_range)

    # Prepare input for model
    grid_points = np.c_[F1.ravel(), F2.ravel()]
    X_rep = np.zeros((grid_points.shape[0], X.shape[1]))  # keep same #features
    X_rep[:, f1_idx] = grid_points[:, 0]
    X_rep[:, f2_idx] = grid_points[:, 1]

    # Predict
    Z = model.predict(X_rep)

    # If multi-output, pick one target
    if Z.ndim > 1:
        Z = Z[:, target_idx]

    Z = Z.reshape(F1.shape)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(F1, F2, Z, cmap="viridis", alpha=0.7)
    ax.scatter(f1, f2, y, color="red", s=20, label="Actual")

    ax.set_xlabel(feature_names[f1_idx])
    ax.set_ylabel(feature_names[f2_idx])
    ax.set_zlabel(f"Predicted Target {target_idx}")
    ax.legend()
    plt.show()
import joblib

# Load the saved TabNet model
tabnet_model = joblib.load("biomedical_rf_model.pkl")   # <-- replace with your actual .pkl filename

# Make sure feature names are available
feature_names = X.columns.tolist()

# Example: plot using first two features against compressive strength (target 0)
plot_tabnet_3d(tabnet_model, X_test, y_test.iloc[:, 0], feature_names, f1_idx=0, f2_idx=1)


# Example: plot using first two features against compressive strength (target 0)
plot_tabnet_3d(tabnet_model, X_test, y_test.iloc[:, 0], feature_names, f1_idx=0, f2_idx=1, target_idx=0)  # Compressive
plot_tabnet_3d(tabnet_model, X_test, y_test.iloc[:, 1], feature_names, f1_idx=0, f2_idx=1, target_idx=1)  # Tensile
plot_tabnet_3d(tabnet_model, X_test, y_test.iloc[:, 2], feature_names, f1_idx=0, f2_idx=1, target_idx=2)  # Flexural

