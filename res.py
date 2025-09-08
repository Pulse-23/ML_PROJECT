import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import xgboost as xgb
from sklearn.model_selection import train_test_split

# =======================
# Global Styling
# =======================
mpl.rcParams['font.family'] = 'Times New Roman'

# Abbreviation dictionary
abbrev_dict = {
    "Biomedical waste ash(kg/m3)": "BMWA(kg/m3)",
    "Biomedical waste ash(%)": "BMWA(%)",
    "Compressive strength (7 days)(MPa)": "CS(7 days)(MPa)",
    "Compressive strength (14 days)(MPa)": "CS(14 days)(MPa)",
    "Compressive strength (28 days)(MPa)": "CS(28 days)(MPa)",
    "Tensile strength(7 days)(MPa)": "TS(7 days)(MPa)",
    "Tensile strength(14 days)(MPa)": "TS(14 days)(MPa)",
    "Tensile strength(28 days)(MPa)": "TS(28 days)(MPa)",
    "Flexural strength(7 days)(MPa)": "FS(7 days)(MPa)",
    "Flexural strength(14 days)(MPa)": "FS(14 days)(MPa)",
    "Flexural strength(28 days)(MPa)": "FS(28 days)(MPa)"
}

# =======================
# Load dataset
# =======================
# Corrected the file path and used pd.read_csv to match the uploaded file
data_path = "D:/ML-civil/Copy of Biomedical waste ash dataset 600(1).xlsx"

df = pd.read_excel(data_path)

# Apply abbreviations
df.rename(columns=abbrev_dict, inplace=True)

# Define targets and features (drop SL.NO if present)
target_columns = ['CS(28 days)(MPa)', 'TS(28 days)(MPa)', 'FS(28 days)(MPa)']
feature_columns = [col for col in df.columns if col not in target_columns and col != "SL.NO"]

X = df[feature_columns]

# Containers to store results for combined plots
train_preds, test_preds, train_residuals, test_residuals, models = {}, {}, {}, {}, {}

# ===========================
# Loop for each target
# ===========================
for target in target_columns:
    print(f"\n================ Analysis for {target} =================")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_preds[target] = y_train_pred
    test_preds[target] = y_test_pred
    train_residuals[target] = y_train - y_train_pred
    test_residuals[target] = y_test - y_test_pred
    models[target] = model

# ===========================
# Combined Train Residuals
# ===========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, target in enumerate(target_columns):
    axes[i].scatter(train_preds[target], train_residuals[target], alpha=0.6,
                    color='blue', s=80)
    axes[i].axhline(y=0, color='red', linestyle='--')
    axes[i].set_title(f"Train Residuals: {target}")
    axes[i].set_xlabel("Predicted Values")
    axes[i].set_ylabel("Residuals")
plt.tight_layout()
plt.show()

# ===========================
# Combined Test Residuals
# ===========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, target in enumerate(target_columns):
    axes[i].scatter(test_preds[target], test_residuals[target], alpha=0.6,
                    color='green', s=130)
    axes[i].axhline(y=0, color='red', linestyle='--')
    axes[i].set_title(f"Test Residuals: {target}")
    axes[i].set_xlabel("Predicted Values")
    axes[i].set_ylabel("Residuals")
plt.tight_layout()
plt.show()

# ===========================
# Separate Feature Importance Plots
# ===========================
for target in target_columns:
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    xgb.plot_importance(models[target],
                        importance_type='weight',
                        xlabel='Score',
                        title=f'Feature Importance for {target}',
                        ax=ax,
                        color='skyblue')

    # Increase font sizes
    ax.set_xlabel("Score", fontsize=14)
    ax.set_ylabel("Features", fontsize=14)
    ax.set_title(f"Feature Importance for {target}", fontsize=15)

    # Remove gridlines
    ax.grid(False)

    # Add border to bars
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    plt.tight_layout()
    plt.show()
