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
data_path = "D:/ML-civil/Copy of Biomedical waste ash dataset 600(1).xlsx"
df = pd.read_excel(data_path)

# Apply abbreviations
df.rename(columns=abbrev_dict, inplace=True)

# Define targets and features (drop SL.NO if present)
target_columns = ['CS(28 days)(MPa)', 'TS(28 days)(MPa)', 'FS(28 days)(MPa)']
feature_columns = [col for col in df.columns if col not in target_columns and col != "SL.NO"]

X = df[feature_columns]

# ===========================
# Loop for each target and plot residuals
# ===========================
for target in target_columns:
    print(f"\n================ Analysis for {target} =================")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    # ===========================
    # Residual plot (Train + Test combined)
    # ===========================
    plt.figure(figsize=(8, 6))

    # Train residuals (blue)
    plt.scatter(y_train_pred, train_residuals,
                alpha=0.6, color='blue', s=80, label="Train")

    # Test residuals (green, different marker for clarity)
    plt.scatter(y_test_pred, test_residuals,
                alpha=0.6, color='green', s=130, marker=".", label="Test")

    # Reference line
    plt.axhline(y=0, color='red', linestyle='--')

    # Titles and labels
    plt.title(f"Residuals: {target}", fontsize=14)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend()

    plt.tight_layout()
    plt.show()
