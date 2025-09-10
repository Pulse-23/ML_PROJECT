import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================
# Paths
# ==============================
model_path = r"D:\ML-civil\saint_model.pkl"
ext_path = r"D:\ML-civil\external_validation_bwa.xlsx"
save_path = r"D:\ML-civil\external_results.xlsx"

# ==============================
# Step 1: Define Model Class
# ==============================
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
        x = self.input_layer(x).unsqueeze(1)   # [batch, 1, embed]
        x = self.transformer(x)                # [batch, 1, embed]
        x = x.mean(dim=1)                      # pool
        return self.fc(x)

# ==============================
# Step 2: Load Model & Scalers
# ==============================
print("Loading model and scalers...")
checkpoint = joblib.load(model_path)
scaler_X = checkpoint["scaler_X"]
scaler_y = checkpoint["scaler_y"]

# rebuild model
input_dim = 4   # cement, bwa, fine agg, coarse agg
output_dim = 3  # compressive, tensile, flexural
model = SAINTModel(input_dim, output_dim)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ==============================
# Step 3: Load External Dataset
# ==============================
print("Loading external validation dataset...")
ext = pd.read_excel(ext_path)

input_cols = ["Cement(kg/m3)", "Biomedical waste ash(kg/m3)",
              "Fine aggregate(kg/m3)", "Coarse aggregate(kg/m3)"]
output_cols = [
    "Compressive strength (28 days)(MPa)",
    "Tensile strength(28 days)(MPa)",
    "Flexural strength(28 days)(MPa)"
]

# Scale inputs
X_ext = scaler_X.transform(ext[input_cols])

# ==============================
# Step 4: Predict with PyTorch
# ==============================
print("Predicting outputs...")
X_tensor = torch.tensor(X_ext, dtype=torch.float32)
with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)

# ==============================
# Step 5: Save Results
# ==============================
results = ext.copy()
for i, col in enumerate(output_cols):
    results[f"Predicted {col}"] = y_pred[:, i]

results.to_excel(save_path, index=False)
print(f"✅ Predictions saved to {save_path}")
# ==============================
# Step 6: Optional Evaluation
# ==============================
metrics_dict = {}

if all(col in ext.columns for col in output_cols):
    # If external dataset has ground truth
    y_true = ext[output_cols].values
    y_true_scaled = scaler_y.transform(y_true)

    # Compute metrics
    mse_loss = nn.MSELoss()(torch.tensor(y_pred_scaled), torch.tensor(y_true_scaled)).item()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    accuracy = 100 - mape

    metrics_dict = {
        "Loss (MSE scaled)": [mse_loss],
        "R² Score": [r2],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE (%)": [mape],
        "Accuracy (%)": [accuracy]
    }

    print("\n📊 External Validation Metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v[0]:.4f}")

else:
    print("\n⚠️ Ground truth not available – only predictions generated.")
# ==============================
# Step 7: Save Predictions + Metrics
# ==============================
import os
if os.path.exists(save_path):
    os.remove(save_path)

with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
    results.to_excel(writer, sheet_name="Predictions", index=False)

    if metrics_dict:
        pd.DataFrame(metrics_dict).to_excel(writer, sheet_name="Metrics", index=False)

print(f"✅ Predictions and metrics saved to {save_path}")