import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from tensorflow import keras


# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_excel("C:/Users/DELL/Documents/ml project/Biomedical waste ash dataset 600.xlsx", sheet_name="Sheet1")
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

# Predictions from NN
nn_preds = nn_model.predict(X_test_scaled)


# ===============================
# 3. LightGBM Models (one per target)
# ===============================
lgb_preds_list = []
for col in y.columns:
    lgb_model = lgb.LGBMRegressor(objective='regression',
                                  n_estimators=500,
                                  learning_rate=0.05,
                                  random_state=42)
    lgb_model.fit(X_train, y_train[col])
    preds = lgb_model.predict(X_test)
    lgb_preds_list.append(preds)

lgb_preds = np.vstack(lgb_preds_list).T  # shape (n_samples, 3)


# ===============================
# 4. Ensemble (Average Predictions)
# ===============================
final_preds = (nn_preds + lgb_preds) / 2


# ===============================
# 5. Extra Metrics
# ===============================
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    acc = 100 * (1 - (mae / np.mean(y_true)))   # accuracy %
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    # Nash-Sutcliffe Efficiency (NS)
    ns = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    # Legates & McCabe Index (LMI)
    lmi = 1 - (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true))))

    # Variance Accounted For (VAF)
    vaf = 100 * (1 - (np.var(y_true - y_pred) / np.var(y_true)))

    # RSR
    rsr = rmse / np.std(y_true)

    return mse, mae, rmse, r2, acc, wmape, ns, lmi, vaf, rsr


# ===============================
# 6. Evaluate per target + overall
# ===============================
results = []
print("\n=== Ensemble Model Results (NN + LightGBM) ===")

for i, col in enumerate(y.columns):
    metrics = calculate_metrics(y_test.iloc[:, i].values, final_preds[:, i])
    mse, mae, rmse, r2, acc, wmape, ns, lmi, vaf, rsr = metrics

    print(f"\nTarget: {col}")
    print(f"  MSE   : {mse:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  R2    : {r2:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  WMAPE : {wmape:.2f}%")
    print(f"  NS    : {ns:.4f}")
    print(f"  LMI   : {lmi:.4f}")
    print(f"  VAF   : {vaf:.2f}%")
    print(f"  RSR   : {rsr:.4f}")

    results.append([col, mse, rmse, r2, mae, acc, wmape, ns, lmi, vaf, rsr])

# Overall averages
overall = np.mean(np.array(results)[:, 1:].astype(float), axis=0)
results.append(["Overall Avg", *overall])

# Convert to DataFrame for clean display
metrics_df = pd.DataFrame(results, columns=["Target", "MSE", "RMSE", "R2", "MAE",
                                            "Accuracy(%)", "WMAPE(%)", "NS", "LMI", "VAF(%)", "RSR"])

print("\n=== Summary Table ===")
print(metrics_df.to_string(index=False))
