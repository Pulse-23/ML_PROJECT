import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset (same file as training, or a new unseen test dataset)
data = pd.read_excel('C:/Users/DELL/Documents/ml project/Biomedical waste ash dataset 600.xlsx', sheet_name='Sheet1')

# Strip spaces from column names
data.columns = data.columns.str.strip()

# Define inputs and outputs
X = data[['Cement(kg/m3)', 'Biomedical waste ash(kg/m3)', 'Fine aggregate(kg/m3)', 'Coarse aggregate(kg/m3)']]
Y = data[['Compressive strength (28 days)(MPa)', 
          'Tensile strength(28 days)(MPa)', 
          'Flexural strength(28 days)(MPa)']]

# Load the trained model
model = load_model("concrete_strength_ann_model.keras")

# ⚠️ Important: Use the same scaler parameters from training
# For real deployment, save & load scalers using joblib. 
# Here, we refit them on the whole dataset for demonstration.
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)

# Predict
Y_pred_scaled = model.predict(X_scaled)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

# Evaluate results
target_names = Y.columns
print("\n=== Testing Results on Dataset ===")
for i, target in enumerate(target_names):
    mse = mean_squared_error(Y.iloc[:, i], Y_pred[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y.iloc[:, i], Y_pred[:, i])
    r2 = r2_score(Y.iloc[:, i], Y_pred[:, i])
    accuracy = 1 - (mae / np.mean(np.abs(Y.iloc[:, i])))

    print(f"\nTarget: {target}")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R2: {r2:.4f}")
    print(f"   Accuracy: {accuracy*100:.2f}%")

# Overall evaluation
mse_overall = mean_squared_error(Y, Y_pred)
rmse_overall = np.sqrt(mse_overall)
mae_overall = mean_absolute_error(Y, Y_pred)
r2_overall = r2_score(Y, Y_pred)
accuracy_overall = 1 - (mae_overall / np.mean(np.abs(Y.values)))

print("\n=== Overall Testing Results ===")
print(f"Overall MSE: {mse_overall:.4f}")
print(f"Overall RMSE: {rmse_overall:.4f}")
print(f"Overall MAE: {mae_overall:.4f}")
print(f"Overall R2 Score: {r2_overall:.4f}")
print(f"Overall Accuracy: {accuracy_overall*100:.2f}%")
