import numpy as np
import matplotlib.pyplot as plt

# ✅ Set global font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# --- Data for Taylor Diagrams ---
stats_train = {
    'Random Forest': [0.820033, 1.603333, 0.820033, 0.185067, 82.00333, 0.991967, 0.389133, 0.1442],
    'SAINT': [0.798000, 1.760000, 0.999700, 0.261200, 75.694590, 1.002600, 0.017600, 0.1643],
    'Tabnet': [0.820033, 1.603000, 0.820033, 0.185067, 82.003467, 0.892100, 0.389133, 0.1442],
    'Ensemble': [0.7244, 0.0208, 0.7244, 0.2510, 76.5078, 0.7350, 0.4835, 0.2038],
}

stats_test = {
    'Random Forest': [0.763800, 1.910000, 0.763800, 0.233033, 76.870000, 0.990433, 0.450200, 0.1852],
    'SAINT': [0.783900, 1.820000, 0.999600, 0.284300, 77.182561, 1.000400, 0.019200, 0.1780],
    'Tabnet': [0.763800, 1.910000, 0.763800, 0.233033, 76.870000, 0.990433, 0.450200, 0.1852],
    'Ensemble': [0.713174, 2.100000, 0.713200, 0.256270, 75.570000, 0.539300, 0.494856, 0.2063],
}

def create_taylor_diagram(ax, stats, title, base_font=15):
    models = list(stats.keys())
    
    r2_values = [stats[m][0] for m in models]
    rsr_values = [stats[m][6] for m in models]
    
    std_obs = stats[models[0]][3] / stats[models[0]][6]
    r_values = [np.sqrt(r2) for r2 in r2_values]
    std_models = [rsr * std_obs for rsr in rsr_values]

    colors = ['b', 'c', 'g', 'r']
    offset_train = {'Random Forest': 0.001, 'Tabnet': -0.001}
    
    for i, model in enumerate(models):
        theta_offset = offset_train.get(model, 0)
        theta = np.arccos(r_values[i]) + theta_offset
        radius = std_models[i]
        ax.plot(theta, radius, 'o', label=model, color=colors[i], markersize=8)

    ax.plot(0, std_obs, 'o', label="Reference", color='k', markersize=8, linestyle='None')
    
    t = np.linspace(0, np.pi/2, 100)
    ax.plot(t, np.ones_like(t) * std_obs, 'k--', label="Observed Std. Dev.")

    ax.set_rmax(std_obs * 1.5)
    ax.set_rticks(np.arange(0, std_obs * 1.5, std_obs / 3))
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_title(title, fontsize=base_font, pad=15)

    # ✅ Legend smaller font
    ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left', fontsize=int(base_font*0.75))

    # ✅ Correlation ticks
    corr_labels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    corr_angles = np.arccos(corr_labels)
    ax.set_xticks(corr_angles)
    ax.set_xticklabels([f'{c:.2f}' for c in corr_labels], fontsize=int(base_font*0.75))

    # ✅ Radial ticks
    ax.set_yticklabels([f"{t:.2f}" for t in ax.get_yticks()], fontsize=int(base_font*0.75))

# --- Main Execution ---
fig, axs = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(12, 5))

create_taylor_diagram(axs[0], stats_train, "Taylor Diagram (Train Set)")
create_taylor_diagram(axs[1], stats_test, "Taylor Diagram (Test Set)")

plt.tight_layout()
plt.show()
