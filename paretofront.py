import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Load Dataset
# ==============================
file_path = "D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")

# Check column names
print("Columns in dataset:", data.columns.tolist())

# ==============================
# Pareto Function
# ==============================
def is_pareto_efficient(costs):
    """
    Identify Pareto-efficient points.
    Args:
        costs: 2D numpy array with [BMWA%, -Property] (minimize BMWA, maximize property)
    Returns:
        Boolean mask for Pareto points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

# ==============================
# Function to Generate Pareto Plot
# ==============================
def generate_pareto_plot(x, y, y_label, title, point_color, pareto_color, filename):
    """
    Generates a Pareto front plot for given parameters.
    """
    # Pareto calculation
    costs = np.column_stack((x, -y))  # Minimize x, Maximize y
    pareto_mask = is_pareto_efficient(costs)
    pareto_points = data.loc[pareto_mask, ["Biomedical waste ash(%)", y_label]].sort_values(by="Biomedical waste ash(%)")

    print(f"\nPareto Optimal Points for {y_label}:\n", pareto_points)

    # Plot
    plt.figure(figsize=(9, 7))
    plt.rcParams["font.family"] = "Times New Roman"

    # All points
    plt.scatter(
        x, y,
        label="All Mixes",
        alpha=0.7,
        color=point_color,
        edgecolor="black",
        s=70
    )

    # Pareto optimal points
    plt.scatter(
        pareto_points["Biomedical waste ash(%)"],
        pareto_points[y_label],
        color=pareto_color,
        label="Pareto Optimal",
        s=120,
        marker="D",
        edgecolor="black"
    )

    # Pareto line
    plt.plot(
        pareto_points["Biomedical waste ash(%)"],
        pareto_points[y_label],
        color=pareto_color,
        linestyle="--",
        linewidth=2,
        label="Pareto Front"
    )

    # Titles & Labels
    plt.title(f"Pareto Front: BMWA% vs {title}", fontsize=18, weight="bold", pad=20)
    plt.xlabel("Biomedical Waste Ash (%)", fontsize=15, labelpad=10)
    plt.ylabel(f"{title} (MPa)", fontsize=15, labelpad=10)

    # Grid & Legend
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="best", frameon=True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(filename, dpi=300)
    plt.show()

# ==============================
# Generate Plots for Each Property
# ==============================
x = data["Biomedical waste ash(%)"]

# 1. Compressive Strength
generate_pareto_plot(
    x=x,
    y=data["Compressive strength (28 days)(MPa)"],
    y_label="Compressive strength (28 days)(MPa)",
    title="Compressive Strength (28 Days)",
    point_color="#FFA500",   # Orange
    pareto_color="#B22222",  # Firebrick Red
    filename="Pareto_Compressive.png"
)

# 2. Flexural Strength
generate_pareto_plot(
    x=x,
    y=data["Flexural strength(28 days)(MPa)"],
    y_label="Flexural strength(28 days)(MPa)",
    title="Flexural strength(28 days)",
    point_color="#20B2AA",   # Light Sea Green
    pareto_color="#006400",  # Dark Green
    filename="Pareto_Flexural.png"
)

# 3. Tensile Strength
generate_pareto_plot(
    x=x,
    y=data["Tensile strength(28 days)(MPa)"],
    y_label="Tensile strength(28 days)(MPa)",
    title="Tensile strength(28 days)",
    point_color="#87CEFA",   # Light Sky Blue
    pareto_color="#00008B",  # Dark Blue
    filename="Pareto_Tensile.png"
)
