import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# -------------------------
# 1. Load trained SAINT model + scalers
# -------------------------
ckpt = joblib.load(r"D:/ML-civil/saint_model.pkl")   # fixed path
scaler_X = ckpt["scaler_X"]
scaler_y = ckpt["scaler_y"]

# -------------------------
# 2. Define SAME model class as training
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
        x = self.input_layer(x).unsqueeze(1)   # add sequence dimension
        x = self.transformer(x)
        x = x.mean(dim=1)                      # pooling
        return self.fc(x)

# -------------------------
# 3. Reload trained model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim, output_dim = 4, 3
model = SAINTModel(input_dim, output_dim).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# -------------------------
# 4. Load external dataset
# -------------------------
ext_path = r"D:/ML-civil/Copy of Biomedical waste ash dataset 600(1).xlsx"  # external file
ext = pd.read_excel(ext_path)

# -------------------------
# 5. Define columns
# -------------------------
input_cols = [
    "Biomedical waste ash(kg/m3)",
    "Cement(kg/m3)",
    "Fine aggregate(kg/m3)",
    "Coarse aggregate(kg/m3)"
]

output_cols = [
    "Compressive strength (28 days)(MPa)",
    "Tensile strength(28 days)(MPa)",
    "Flexural strength(28 days)(MPa)"
]

# -------------------------
# 6. Prediction function
# -------------------------
def predict_model(X):
    with torch.no_grad():
        X_scaled = scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        y_scaled = model(X_tensor).cpu().numpy()
        y = scaler_y.inverse_transform(y_scaled)
    return y

# -------------------------
# 7. Run external validation
# -------------------------
X_ext = ext[input_cols].values
y_ext_true = ext[output_cols].values
y_ext_pred = predict_model(X_ext)

# Save predictions
results = pd.DataFrame(y_ext_pred, columns=[col + "_Pred" for col in output_cols])
final_df = pd.concat([ext.reset_index(drop=True), results], axis=1)

# -------------------------
# 8. Export results
# -------------------------
final_df.to_excel(r"D:/ML-civil/external_validation_results.xlsx", index=False)
print("✅ External validation results saved at D:/ML-civil/external_validation_results.xlsx")
