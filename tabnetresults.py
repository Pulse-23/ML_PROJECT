import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pytorch_tabnet.tab_model import TabNetRegressor

# === Load your dataset ===
# Example: df = pd.read_csv("yourdata.csv")
# Load dataset
df = pd.read_excel("D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx")
# Suppose last column is target, first columns are features
target_columns = ['Compressive strength (28 days)(MPa)', 
        'Tensile strength(28 days)(MPa)', 
        'Flexural strength(28 days)(MPa)']
X = df.drop(columns=["target_column"]).values
y = df["target_column"].values
# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optional: Scaling (helps TabNet sometimes)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Custom Regression Metrics ===
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

# === Function to evaluate metrics ===
def evaluate_metrics(y_true, y_pred, dataset_name=""):
    return {
        "Dataset": dataset_name,
        "R2": round(r2_score(y_true, y_pred), 4),
        "WAPE": round(wape(y_true, y_pred), 4),
        "NS": round(nash_sutcliffe(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "VAF": round(vaf(y_true, y_pred), 4),
        "LMI": round(lin_concordance_corr(y_true, y_pred), 4),
        "RSR": round(rsr(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
    }

# === Train TabNet ===
model = TabNetRegressor()
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=["train", "valid"],
    eval_metric=["r2"],
    max_epochs=100,
    patience=10,
    batch_size=256,
    virtual_batch_size=128,
    verbose=1
)

# === Predictions ===
y_train_pred = model.predict(X_train).flatten()
y_test_pred = model.predict(X_test).flatten()

# === Build Results Table ===
train_results = evaluate_metrics(y_train, y_train_pred, "Train")
test_results = evaluate_metrics(y_test, y_test_pred, "Test")

results_df = pd.DataFrame([train_results, test_results])
print("\n=== TabNet Results Table ===")
print(results_df)
