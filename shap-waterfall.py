import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Global style (Times New Roman, fontsize 15 everywhere)
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15   # ✅ set default font size

# Load dataset
data_path = "D:/ML-civil/Copy of Biomedical waste ash dataset 600(1).xlsx"
df = pd.read_excel(data_path)

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

# Rename columns
df.rename(columns=abbrev_dict, inplace=True)

# Drop unwanted features
for col in ["SL.NO", "Fine aggregate", "Coarse aggregate"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Target and features
target = "CS(28 days)(MPa)"
X = df.drop(columns=[target])
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# SHAP explainer and values
explainer = shap.Explainer(model, X_train, feature_names=X.columns)
shap_values = explainer(X_test)

# Custom feature order
custom_order = [
    "Cement(kg/m3)",
    "BMWA(kg/m3)",
    "BMWA(%)",
    "CS(7 days)(MPa)",
    "CS(14 days)(MPa)",
    "CS(28 days)(MPa)",
    "TS(7 days)(MPa)",
    "TS(14 days)(MPa)",
    "TS(28 days)(MPa)",
    "FS(7 days)(MPa)",
    "FS(14 days)(MPa)",
    "FS(28 days)(MPa)"
]

# Get first sample SHAP values and feature names
sample_idx = 0
vals = shap_values[sample_idx].values
names = shap_values[sample_idx].feature_names

# Create dictionary for quick index lookup
name_to_val = dict(zip(names, vals))

# Order values and names by custom_order, fill missing with 0
ordered_vals = [name_to_val.get(f, 0) for f in custom_order]
ordered_names = custom_order

# Colors: blue for positive, red for negative
colors = ['blue' if val >= 0 else 'red' for val in ordered_vals]

# Plot mimic waterfall as horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 6))
y_pos = np.arange(len(ordered_names))
ax.barh(y_pos, ordered_vals, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(ordered_names, fontsize=15)  # ✅ fontsize 15
ax.axvline(0, color='k', linestyle='--')
ax.set_xlabel("SHAP value (impact on prediction)", fontsize=15)  # ✅ fontsize 15
ax.tick_params(axis='x', labelsize=15)  # ✅ x-ticks fontsize
ax.tick_params(axis='y', labelsize=15)  # ✅ y-ticks fontsize
ax.invert_yaxis()  # highest impact on top

# Add SHAP value labels on the bars
for i, val in enumerate(ordered_vals):
    ax.text(val + (0.02 if val >= 0 else -0.02), i, f"{val:+.2f}",
            va='center',
            ha='left' if val >= 0 else 'right',
            color='black',
            fontsize=15)  # ✅ fontsize 15

plt.tight_layout()
plt.show()
