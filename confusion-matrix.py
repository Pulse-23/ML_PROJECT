import pandas as pd
import numpy as np
import joblib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tensorflow import keras

# ===========================
# 1. Load dataset
# ===========================
file_path = r"D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx"
df = pd.read_excel(file_path)

X = df[['Cement(kg/m3)', 'Biomedical waste ash(kg/m3)',
        'Fine aggregate(kg/m3)', 'Coarse aggregate(kg/m3)']]

y = df[['Compressive strength (28 days)(MPa)',
        'Tensile strength(28 days)(MPa)',
        'Flexural strength(28 days)(MPa)']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# 2. SAINT model definition
# ===========================
class SAINTModel(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim=64, depth=2, heads=4):
        super(SAINTModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads,
                                                   dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ===========================
# 3. Load SAINT
# ===========================
saint_info = joblib.load("saint_model.pkl")
model_state = saint_info["model_state"]
scaler_X_saint = saint_info["scaler_X"]
scaler_y_saint = saint_info["scaler_y"]

saint_model = SAINTModel(input_dim=4, output_dim=3)
saint_model.load_state_dict(model_state)
saint_model.eval()

X_test_scaled = scaler_X_saint.transform(X_test)
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
with torch.no_grad():
    saint_preds = saint_model(X_tensor).numpy()
saint_preds = scaler_y_saint.inverse_transform(saint_preds)

# ===========================
# 4. Random Forest
# ===========================
rf_model = joblib.load("random_forest_model.pkl")
rf_preds = rf_model.predict(X_test)

# ===========================
# 5. TabNet (actually RF)
# ===========================
tabnet_model = joblib.load("biomedical_rf_model.pkl")
tabnet_preds = tabnet_model.predict(X_test)

# ===========================
# 6. Ensemble
# ===========================
with open("ensemble_config.pkl", "rb") as f:
    ens_info = pickle.load(f)

# Load NN + scaler
nn_model = keras.models.load_model(ens_info["nn_model"], compile=False)
nn_scaler = joblib.load(ens_info["nn_scaler"])
X_test_nn = nn_scaler.transform(X_test)
nn_preds = nn_model.predict(X_test_nn)

# Load LightGBM models
lgb_preds_list = []
for col in ens_info["lgb_models"]:
    fname = f"lgb_model_{col.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    lgb_model = joblib.load(fname)
    preds = lgb_model.predict(X_test)
    lgb_preds_list.append(preds)
lgb_preds = np.vstack(lgb_preds_list).T

ensemble_preds = (nn_preds + lgb_preds) / 2

# ===========================
# 7. Collect results
# ===========================
models = {
    "SAINT": saint_preds,
    "Random Forest": rf_preds,
    "TabNet": tabnet_preds,
    "Ensemble": ensemble_preds
}

targets = ['Compressive strength (28 days)(MPa)',
           'Tensile strength(28 days)(MPa)',
           'Flexural strength(28 days)(MPa)']

# ===========================
# 8. Plot correlation heatmaps
# ===========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 17   # base size

fig, axes = plt.subplots(len(models), len(targets), figsize=(20, 14))

for row, (model_name, preds) in enumerate(models.items()):
    for col, target in enumerate(targets):
        actual = y_test[target].values
        predicted = preds[:, col]

        # Correlation matrix
        corr_matrix = np.corrcoef(actual, predicted)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=["Actual", model_name],
                    yticklabels=["Actual", model_name],
                    ax=axes[row, col],
                    cbar=True,
                    annot_kws={"size": 15})

        if row == 0:
            axes[row, col].set_title(target, fontsize=20, fontweight="bold")

        # Add vertical model label on the left side
        if col == 0:
            axes[row, col].set_ylabel(model_name, fontsize=20, fontweight="bold", rotation=90)

plt.tight_layout()
plt.show()
