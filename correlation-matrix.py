import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Drop "Sl.no" if it exists
df_corr = df.drop(columns=["SL.NO", "Fine aggregate(kg/m3)", "Coarse aggregate(kg/m3)"], errors="ignore")
# Rename outputs to abbreviations
rename_dict = {
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
df_corr.rename(columns=rename_dict, inplace=True)
# Compute correlation matrix
corr_matrix = df_corr.corr()

# Create mask for upper triangle (keep diagonal visible)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    square=True,
    fmt=".2f",
    mask=mask,
    annot_kws={"fontname": "Times New Roman", "fontsize": 10},
    cbar_kws={"shrink": 0.8}
)

# Title and font styling
plt.title("Correlation Matrix of Dataset Features",
          fontsize=14, fontweight="bold", fontname="Times New Roman")

plt.xticks(fontsize=10, fontname="Times New Roman", rotation=45, ha="right")
plt.yticks(fontsize=10, fontname="Times New Roman", rotation=0)

plt.tight_layout()
plt.show()

