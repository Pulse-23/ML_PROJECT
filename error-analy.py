import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===========================
# Set Global Font to Times New Roman
# ===========================
plt.rcParams["font.family"] = "Times New Roman"

# ===========================
# Metrics Data (Train + Test)
# ===========================
metrics_train = {
    "Random Forest": {"R2": 0.820033, "WMAPE": 1.603333, "NS": 0.820033, "RMSE": 0.185067,
                      "VAF": 82.003333, "LMI": 0.991967, "RSR": 0.389133, "MAE": 0.1442},
    "SAINT": {"R2": 0.798000, "WMAPE": 1.760000, "NS": 0.999700, "RMSE": 0.261200,
              "VAF": 75.694590, "LMI": 1.002600, "RSR": 0.017600, "MAE": 0.1643},
    "Tabnet": {"R2": 0.820033, "WMAPE": 1.603000, "NS": 0.820033, "RMSE": 0.185067,
               "VAF": 82.003467, "LMI": 0.892100, "RSR": 0.389133, "MAE": 0.1442},
    "Ensemble": {"R2": 0.7244, "WMAPE": 2.08000, "NS": 0.7244, "RMSE": 0.2510,
                 "VAF": 76.5078, "LMI": 0.7350, "RSR": 0.4835, "MAE": 0.2038},
}

metrics_test = {
    "Random Forest": {"R2": 0.763800, "WMAPE": 1.910000, "NS": 0.763800, "RMSE": 0.233033,
                      "VAF": 76.870000, "LMI": 0.990433, "RSR": 0.450200, "MAE": 0.1852},
    "SAINT": {"R2": 0.783900, "WMAPE": 1.820000, "NS": 0.999600, "RMSE": 0.284300,
              "VAF": 77.182561, "LMI": 1.000400, "RSR": 0.019200, "MAE": 0.1780},
    "Tabnet": {"R2": 0.763800, "WMAPE": 1.910000, "NS": 0.763800, "RMSE": 0.233033,
               "VAF": 76.870000, "LMI": 0.990433, "RSR": 0.450200, "MAE": 0.1852},
    "Ensemble": {"R2": 0.713174, "WMAPE": 2.100000, "NS": 0.713200, "RMSE": 0.256270,
                 "VAF": 75.570000, "LMI": 0.539300, "RSR": 0.494856, "MAE": 0.2063},
}

# ===========================
# Plotting Function
# ===========================
def plot_error_analysis(metrics_train, metrics_test):
    models = list(metrics_train.keys())
    metrics = list(metrics_train[models[0]].keys())

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))  # 8 metrics in 2 rows x 4 cols
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        train_vals = [metrics_train[m][metric] for m in models]
        test_vals = [metrics_test[m][metric] for m in models]

        x = np.arange(len(models))
        width = 0.35

        axes[i].bar(x - width/2, train_vals, width, label="Train", alpha=0.7,
                    edgecolor="black", linewidth=0.7)
        axes[i].bar(x + width/2, test_vals, width, label="Test", alpha=0.7,
                    edgecolor="black", linewidth=0.7)

        # Titles + ticks (25% bigger)
        axes[i].set_title(metric, fontsize=int(12 * 1.25))
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=30, fontsize=int(10 * 1.25))
        axes[i].tick_params(axis="y", labelsize=int(10 * 1.25))

        # Legend (keep original size)
        axes[i].legend(fontsize=8)

    plt.title("Error Analysis: Train vs Test", fontsize=int(14 * 1.25), y=1.02)
    plt.tight_layout()
    plt.show()


# ===========================
# Run Plot
# ===========================
plot_error_analysis(metrics_train, metrics_test)
