import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt

# Global style (Times New Roman, fontsize 15 everywhere)
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15

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

# Drop unwanted columns
df = df.drop(columns=["SL.NO", "Fine aggregate", "Coarse aggregate"], errors='ignore')

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

# Plot waterfall for first sample
shap.plots.waterfall(shap_values[0], max_display=12, show=True)
