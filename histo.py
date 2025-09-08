import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# =======================
# Global Font Setting
# =======================
mpl.rcParams['font.family'] = 'Times New Roman'

# =======================
# Load dataset
# =======================
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

# Apply abbreviations
df.rename(columns=abbrev_dict, inplace=True)

# =======================
# Target columns
# =======================
targets = ["CS(28 days)(MPa)", "TS(28 days)(MPa)", "FS(28 days)(MPa)"]

# =======================
# Plot each separately
# =======================
for target in targets:
    plt.figure(figsize=(6, 5))
    sns.histplot(df[target], bins=20, kde=True,
                 color="navy", edgecolor="black", alpha=0.5)
    plt.title(f"Histogram of {target}", fontsize=15)
    plt.xlabel(target, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.show()
