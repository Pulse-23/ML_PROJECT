import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# 1. Global Styling
# ==============================
plt.rcParams['font.family'] = 'Times New Roman'  # Set font to Times New Roman
plt.rcParams['font.size'] = 14                  # Base font size
plt.rcParams['axes.titlesize'] = 18             # Title font size
plt.rcParams['axes.labelsize'] = 16             # Axis label font size
plt.rcParams['xtick.labelsize'] = 14            # X-tick size
plt.rcParams['ytick.labelsize'] = 14            # Y-tick size

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_excel("C:/Users/DELL/Documents/ml project/Biomedical waste ash dataset 600.xlsx")

# ==============================
# 3. Boxplots with Custom Colors
# ==============================
# Color palette for better visibility
palette = sns.color_palette("viridis")

# --- Compressive Strength ---
plt.figure(figsize=(9, 6))
sns.boxplot(x='Biomedical waste ash(%)', y='Compressive strength (28 days)(MPa)', 
            data=df, palette=palette)
plt.title('Compressive Strength (28 days) by Biomedical Waste Ash (%)', color='darkblue', fontsize=20)
plt.ylabel('Compressive Strength (MPa)', color='black')
plt.xlabel('Biomedical Waste Ash (%)', color='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Tensile Strength ---
plt.figure(figsize=(9, 6))
sns.boxplot(x='Biomedical waste ash(%)', y='Tensile strength(28 days)(MPa)', 
            data=df, palette=palette)
plt.title('Tensile Strength (28 days) by Biomedical Waste Ash (%)', color='darkblue', fontsize=20)
plt.ylabel('Tensile Strength (MPa)', color='black')
plt.xlabel('Biomedical Waste Ash (%)', color='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Flexural Strength ---
plt.figure(figsize=(9, 6))
sns.boxplot(x='Biomedical waste ash(%)', y='Flexural strength(28 days)(MPa)', 
            data=df, palette=palette)
plt.title('Flexural Strength (28 days) by Biomedical Waste Ash (%)', color='darkblue', fontsize=20)
plt.ylabel('Flexural Strength (MPa)', color='black')
plt.xlabel('Biomedical Waste Ash (%)', color='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
