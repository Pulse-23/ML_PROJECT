import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# -------------------------
# 1. Load dataset
# -------------------------
file_path = "D:/ML-civil/Copy of Biomedical waste ash dataset 600(1).xlsx"
df = pd.read_excel(file_path)

# Define inputs and outputs
X = df[["Biomedical waste ash(%)", "Cement(kg/m3)", "Fine aggregate(kg/m3)",	"Coarse aggregate(kg/m3)"]].values
y = df[["Compressive strength (28 days)(MPa)", "Tensile strength(28 days)(MPa)", "Flexural strength(28 days)(MPa)"]].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize inputs and outputs
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# -------------------------
# 2. PyTorch Dataset
# -------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TabularDataset(X_train, y_train)
test_dataset = TabularDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn

class SAINTModel(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim=64, depth=2, heads=4):
        super(SAINTModel, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: [batch, input_dim]
        x = self.input_layer(x).unsqueeze(1)   # add sequence dim
        x = self.transformer(x)                # transformer expects [batch, seq, embed]
        x = x.mean(dim=1)                      # pool over sequence
        return self.fc(x)

# -------------------------
# 4. Training
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAINTModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
# -------------------------
# 5. Evaluation Metrics
# -------------------------
model.eval()
y_pred_scaled = []
y_true_scaled = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        y_pred_scaled.append(outputs.cpu().numpy())
        y_true_scaled.append(batch_y.cpu().numpy())

y_pred_scaled = np.vstack(y_pred_scaled)
y_true_scaled = np.vstack(y_true_scaled)

# Inverse transform to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_true_scaled)

# Metrics
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
# (1) MAPE-based accuracy
accuracy_mape = 100 - wmape

# (2) Tolerance-based accuracy (within 10% of true value)
tolerance = 0.1
within_tol = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)) < tolerance) * 100

print("\n📊 Final Evaluation Metrics:")
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

# -------------------------
# 6. Save Model & Scalers
# -------------------------
joblib.dump({
    "model_state": model.state_dict(),
    "scaler_X": scaler_X,
    "scaler_y": scaler_y
}, "saint_model.pkl")

print("\n✅ Model and scalers saved as saint_model.pkl")