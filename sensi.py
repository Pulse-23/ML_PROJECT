import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = r"D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Define input and output columns
input_cols = [
    "Cement(kg/m3)",
    "Biomedical waste ash(kg/m3)",
]

output_cols = [
    "Compressive strength (28 days)(MPa)",
    "Tensile strength(28 days)(MPa)",
    "Flexural strength(28 days)(MPa)"
]

X = df[input_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- FONT SETTINGS ---
main_fontsize = 15
legend_fontsize = 8
font_settings = {"fontname": "Times New Roman", "fontsize": main_fontsize}

# Function to plot sensitivity curves for a single target
def get_sensitivity_curves(X_scaled, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    baseline_pred = model.predict(X_test).mean()
    curves = {}

    for i, feature in enumerate(feature_names):
        values = np.linspace(X_scaled[:, i].min(), X_scaled[:, i].max(), 50)
        preds = []
        for val in values:
            X_temp = X_test.copy()
            X_temp[:, i] = val
            preds.append(model.predict(X_temp).mean())
        curves[feature] = (values, preds)

    return curves, baseline_pred

# --- Generate separate plots ---
for target in output_cols:
    y = df[target].values
    curves, baseline_pred = get_sensitivity_curves(X_scaled, y, input_cols)

    plt.figure(figsize=(8, 6))
    for feature, (values, preds) in curves.items():
        plt.plot(values, preds, label=feature, linewidth=2)

    plt.axhline(y=baseline_pred, color='r', linestyle='--', label="Baseline Prediction")

    # Titles and labels
    plt.xlabel("Standardized Feature Value", **font_settings)
    plt.ylabel(f"Predicted {target}", **font_settings)
    plt.title(f"Sensitivity Curve for {target}", **font_settings, pad=10)

    # Tick labels
    plt.tick_params(axis='both', which='major', labelsize=main_fontsize)
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontname("Times New Roman")

    # Legend
    legend = plt.legend(
        prop={"family": "Times New Roman", "size": legend_fontsize},
        handlelength=1.2,
        handleheight=0.6,
        labelspacing=0.4
    )
    for text in legend.get_texts():
        text.set_fontname("Times New Roman")

    plt.tight_layout(pad=2.0)
    plt.show()
