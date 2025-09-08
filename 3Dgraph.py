import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

# -------------------------
# 1. Load dataset
# -------------------------
file_path = "D:\\ML-civil\\Copy of Biomedical waste ash dataset 600(1).xlsx"
df = pd.read_excel(file_path)

# -------------------------
# 2. Define SAME model as training
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
        x = self.input_layer(x).unsqueeze(1)   # add sequence dim
        x = self.transformer(x)
        x = x.mean(dim=1)                      # pool
        return self.fc(x)

# -------------------------
# 3. Load trained model + scalers
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = joblib.load("saint_model.pkl")
scaler_X = checkpoint["scaler_X"]
scaler_y = checkpoint["scaler_y"]

input_dim = 4
output_dim = 3
model = SAINTModel(input_dim, output_dim).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -------------------------
# 4. Prediction function
# -------------------------
def predict_model(X):
    with torch.no_grad():
        X_scaled = scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        y_scaled = model(X_tensor).cpu().numpy()
        y = scaler_y.inverse_transform(y_scaled)
    return y
# -------------------------
# 5. Plotting function (Separate Figures)
# -------------------------
def plot_all_surfaces(fixed_values):
    # Independent variables for 3D plotting
    x_var = "Cement(kg/m3)"
    y_var = "Biomedical waste ash(%)"
    
    x_range = np.linspace(df[x_var].min(), df[x_var].max(), 30)
    y_range = np.linspace(df[y_var].min(), df[y_var].max(), 30)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    # Prepare input combinations
    all_inputs = []
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            vals = fixed_values.copy()
            vals[input_cols.index(x_var)] = X_grid[j, i]
            vals[input_cols.index(y_var)] = Y_grid[j, i]
            all_inputs.append(vals)

    all_inputs = np.array(all_inputs)
    preds = predict_model(all_inputs)

    # ---- Font settings ----
    base_size = 12
    font_settings = {"fontname": "Times New Roman", "fontsize": base_size * 1.25}

    # Short names for outputs
    short_labels = [
        "CS (28 days)(MPa)",
        "TS (28 days)(MPa)",
        "FS (28 days)(MPa)"
    ]

    # Plot each output separately
    for idx, z_label in enumerate(output_cols):
        Z = preds[:, idx].reshape(X_grid.shape)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap="viridis", edgecolor="none")

        ax.set_xlabel("Cement (kg/m3)", **font_settings, labelpad=12)
        ax.set_ylabel("BMWA (%)", **font_settings, labelpad=12)
        ax.set_zlabel(short_labels[idx], **font_settings, labelpad=12)
        ax.set_title(f"{short_labels[idx]} vs Cement & BMWA", **font_settings, pad=20)

        # Ticks also in Times New Roman
        ax.tick_params(axis='both', which='major', labelsize=base_size * 1.25)
        for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            label.set_fontname("Times New Roman")

        # Colorbar
        cbar = fig.colorbar(
            surf, ax=ax, shrink=0.7, aspect=12, pad=0.15, location="right"
        )
        cbar.ax.tick_params(labelsize=base_size * 1.1)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname("Times New Roman")

        plt.tight_layout()
        plt.show()



# -------------------------
# 6. Column names
# -------------------------
input_cols = [
    "Biomedical waste ash(%)",
    "Cement(kg/m3)",
    "Fine aggregate(kg/m3)",
    "Coarse aggregate(kg/m3)"
]

output_cols = [
    "Compressive strength (28 days)(MPa)",
    "Tensile strength (28 days)(MPa)",
    "Flexural strength (28 days)(MPa)"
]

# -------------------------
# 7. Fix unused inputs at mean
# -------------------------
fixed_vals = [
    df["Biomedical waste ash(%)"].mean(),
    df["Cement(kg/m3)"].mean(),
    df["Fine aggregate(kg/m3)"].mean(),
    df["Coarse aggregate(kg/m3)"].mean()
]

# -------------------------
# 8. Call combined plotting function
# -------------------------
plot_all_surfaces(fixed_vals)
