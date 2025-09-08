import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# -------------------------
# 1. Load dataset
# -------------------------
file_path = "D:/ML-civil/Copy of Biomedical waste ash dataset 600(1).xlsx"
df = pd.read_excel(file_path)

# Input and Output columns
input_cols = [
    "Biomedical waste ash(%)",
    "Cement(kg/m3)",
    "Fine aggregate(kg/m3)",
    "Coarse aggregate(kg/m3)"
]

output_cols = [
    "Compressive strength (28 days)(MPa)",
    "Tensile strength(28 days)(MPa)",
    "Flexural strength(28 days)(MPa)"
]

X = df[input_cols].values
y = df[output_cols].values

# -------------------------
# 2. Load saved model + scalers
# -------------------------
class SAINTModel(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim=64, depth=2, heads=4):
        super(SAINTModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x).unsqueeze(1)  # [batch, 1, embed_dim]
        x = self.transformer(x)
        x = x.mean(dim=1)                     # pool sequence dimension
        return self.fc(x)

# Load checkpoint
checkpoint = joblib.load("saint_model.pkl")
scaler_X = checkpoint["scaler_X"]
scaler_y = checkpoint["scaler_y"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAINTModel(input_dim=len(input_cols), output_dim=len(output_cols)).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -------------------------
# 3. Scale data
# -------------------------
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# -------------------------
# 4. Predictions
# -------------------------
with torch.no_grad():
    y_pred_scaled = model(X_tensor).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = y

# -------------------------
# 5. Metrics
# -------------------------
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

wmape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
ns = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
vaf = (1 - np.var(y_true - y_pred) / np.var(y_true)) * 100
lmi = (2 * np.cov(y_true.flatten(), y_pred.flatten())[0, 1]) / (
    np.var(y_true.flatten()) + np.var(y_pred.flatten()) + (np.mean(y_true) - np.mean(y_pred)) ** 2
)
rsr = rmse / np.std(y_true)

# Accuracy definitions
accuracy_mape = 100 - wmape  # 1) MAPE-based accuracy
tolerance = 0.1              # 2) within 10% tolerance
within_tol = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)) < tolerance) * 100

# -------------------------
# 6. Print results
# -------------------------
print("\n📊 Final Evaluation Metrics on Full Dataset:")
print(f"Accuracy (MAPE-based): {accuracy_mape:.2f}%")
print(f"Accuracy (within 10% tolerance): {within_tol:.2f}%")
print(f"R² Score: {r2:.4f}")
print(f"WMAPE: {wmape:.2f}%")
print(f"NS (Nash-Sutcliffe Efficiency): {ns:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"VAF: {vaf:.2f}%")
print(f"LMI: {lmi:.4f}")
print(f"RSR: {rsr:.4f}")
print(f"MAE: {mae:.4f}")
