import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

# Example data
np.random.seed(42)
data = pd.DataFrame({
    'CS': np.random.uniform(20, 80, 200),
    'FS': np.random.uniform(5, 20, 200),
    'TS': np.random.uniform(2, 10, 200),
    'NFA': np.random.uniform(550, 950, 200),
    'NCA': np.random.uniform(50, 1200, 200),
    'RCA': np.random.uniform(50, 1200, 200),
    'PD': np.random.uniform(0.55, 0.82, 200)
})

# Apply Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'

# Scatter + Histogram plotting function
def scatter_with_hist(x, y, color, xlabel, ylabel, title, fig, row, col):
    outer_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[row, col],
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.12, hspace=0.12
    )

    ax_scatter = fig.add_subplot(outer_gs[1, 0])
    ax_scatter.scatter(data[x], data[y], color=color, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    ax_scatter.set_xlabel(xlabel, fontsize=12, labelpad=5)
    ax_scatter.set_ylabel(ylabel, fontsize=12, labelpad=5)
    ax_scatter.set_title(title, fontsize=14, pad=15)

    for spine in ax_scatter.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    ax_histx = fig.add_subplot(outer_gs[0, 0], sharex=ax_scatter)
    ax_histx.hist(data[x], bins=20, color=color, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histx.set_yticks([])

    for spine in ax_histx.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    ax_histy = fig.add_subplot(outer_gs[1, 1], sharey=ax_scatter)
    ax_histy.hist(data[y], bins=20, orientation='horizontal',
                  color=color, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax_histy.tick_params(axis='y', labelleft=False)
    ax_histy.set_xticks([])

    for spine in ax_histy.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)


# ===== Plot 1: Compressive Strength vs Features =====
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Compressive Strength vs Material Properties', fontsize=18, y=0.95)  # Adjusted y-position
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35)

scatter_with_hist('NFA', 'CS', 'red', 'NFA', 'Compressive Strength (CS)', 'CS vs NFA', fig, 0, 0)
scatter_with_hist('NCA', 'CS', 'orange', 'NCA', 'Compressive Strength (CS)', 'CS vs NCA', fig, 0, 1)
scatter_with_hist('RCA', 'CS', 'purple', 'RCA', 'Compressive Strength (CS)', 'CS vs RCA', fig, 1, 0)
scatter_with_hist('PD', 'CS', 'deeppink', 'PD', 'Compressive Strength (CS)', 'CS vs PD', fig, 1, 1)

plt.tight_layout()
plt.show()


# ===== Plot 2: Flexural Strength vs Features =====
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Flexural Strength vs Material Properties', fontsize=18, y=0.95)  # Adjusted y-position
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35)

scatter_with_hist('NFA', 'FS', 'red', 'NFA', 'Flexural Strength (FS)', 'FS vs NFA', fig, 0, 0)
scatter_with_hist('NCA', 'FS', 'orange', 'NCA', 'Flexural Strength (FS)', 'FS vs NCA', fig, 0, 1)
scatter_with_hist('RCA', 'FS', 'purple', 'RCA', 'Flexural Strength (FS)', 'FS vs RCA', fig, 1, 0)
scatter_with_hist('PD', 'FS', 'deeppink', 'PD', 'Flexural Strength (FS)', 'FS vs PD', fig, 1, 1)

plt.tight_layout()
plt.show()


# ===== Plot 3: Tensile Strength vs Features =====
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Tensile Strength vs Material Properties', fontsize=18, y=0.95)  # Adjusted y-position
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35)

scatter_with_hist('NFA', 'TS', 'red', 'NFA', 'Tensile Strength (TS)', 'TS vs NFA', fig, 0, 0)
scatter_with_hist('NCA', 'TS', 'orange', 'NCA', 'Tensile Strength (TS)', 'TS vs NCA', fig, 0, 1)
scatter_with_hist('RCA', 'TS', 'purple', 'RCA', 'Tensile Strength (TS)', 'TS vs RCA', fig, 1, 0)
scatter_with_hist('PD', 'TS', 'deeppink', 'PD', 'Tensile Strength (TS)', 'TS vs PD', fig, 1, 1)

plt.tight_layout()
plt.show()
