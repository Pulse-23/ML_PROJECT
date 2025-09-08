import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from tensorflow import keras
import joblib  # for saving models/scaler
import pickle  # for saving ensemble config

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_excel(r"D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx", sheet_name="Sheet1")
df = df.drop(columns=["SL.NO"], errors="ignore")

# Features and Targets
X = df[["Cement(kg/m3)", "Biomedical waste ash(kg/m3)",
        "Fine aggregate(kg/m3)", "Coarse aggregate(kg/m3)"]]

y = df[["Compressive strength (28 days)(MPa)",
        "Tensile strength(28 days)(MPa)",
        "Flexural strength(28 days)(MPa)"]]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize for NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 2. Neural Network Model
# ===============================
nn_model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3)  # 3 outputs
])

nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train_scaled, y_train.values, epochs=200, batch_size=16, verbose=0)

# Save NN model + scaler
nn_model.save("nn_model.h5")
joblib.dump(scaler, "nn_scaler.pkl")

# ===============================
# 3. LightGBM Models (one per target)
# ===============================
lgb_models = {}
lgb_preds_list = []
for col in y.columns:
    lgb_model = lgb.LGBMRegressor(objective='regression',
                                  n_estimators=500,
                                  learning_rate=0.05,
                                  random_state=42)
    lgb_model.fit(X_train, y_train[col])
    preds = lgb_model.predict(X_test)
    lgb_preds_list.append(preds)
    
    # Save each LightGBM model separately
    lgb_models[col] = lgb_model
    joblib.dump(lgb_model, f"lgb_model_{col.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")

lgb_preds = np.vstack(lgb_preds_list).T  # shape (n_samples, 3)

# ===============================
# 4. Ensemble (Average Predictions)
# ===============================
nn_preds = nn_model.predict(X_test_scaled)
final_preds = (nn_preds + lgb_preds) / 2

# Save ensemble config for later use
with open("ensemble_config.pkl", "wb") as f:
    pickle.dump({
        "nn_model": "nn_model.h5",
        "nn_scaler": "nn_scaler.pkl",
        "lgb_models": list(y.columns)  # ordered by targets
    }, f)

# ===============================
# 5. Metrics Calculation
# ===============================
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    ns = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    vaf = (1 - np.var(y_true - y_pred) / np.var(y_true)) * 100
    lmi = 1 - (rmse / (np.std(y_true) + np.std(y_pred)))
    rsr = rmse / np.std(y_true)
    
    return {
        "R2": r2,
        "WMAPE": wmape,
        "NS": ns,
        "RMSE": rmse,
        "VAF (%)": vaf,
        "LMI": lmi,
        "RSR": rsr,
        "MAE": mae
    }

# ===============================
# 6. Collect Metrics per Target
# ===============================
results = []
for i, col in enumerate(y.columns):
    metrics = compute_metrics(y_test.iloc[:, i].values, final_preds[:, i])
    results.append(metrics)

# Convert to DataFrame
results_df = pd.DataFrame(results, index=y.columns)

# ===============================
# 7. Overall Metrics (average across targets)
# ===============================
overall_metrics = {k: np.mean([res[k] for res in results]) for k in results[0].keys()}
results_df.loc["Overall"] = overall_metrics

# ===============================
# 8. Display Final Metrics Table
# ===============================
print("\n=== Overall Metrics Table ===")
print(results_df.round(4))
