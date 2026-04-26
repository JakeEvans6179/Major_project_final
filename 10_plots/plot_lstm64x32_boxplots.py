import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =========================
# Load results
# =========================
federated = pd.read_csv("federated_ft_lstm64x32.csv")

# FL before and after fine-tuning
fed = federated["federated_mean_rmse"].dropna().values
fed_ft = federated["fine_tuned_mean_rmse"].dropna().values

# =========================
# Plot settings
# =========================
data = [fed, fed_ft]
labels = ["FL", "FL + FT"]
colors = ["#E6C06B", "#8CCB8A"]   # warm yellow, soft green

fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)

box = ax.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    widths=0.1,
    medianprops=dict(color="black", linewidth=1.5),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.0),
    capprops=dict(linewidth=1.0),
    flierprops=dict(
        marker='o',
        markersize=4,
        markerfacecolor='white',
        markeredgecolor='black',
        alpha=0.9
    )
)

# Fill box colours
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# =========================
# Labels and styling
# =========================
ax.set_title("LSTM64x32 RMSE Before and After Fine-Tuning", fontsize=12, pad=10)
ax.set_xlabel("Training Regime", fontsize=10)
ax.set_ylabel("Household Mean RMSE (kWh)", fontsize=10)

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Optional: annotate medians
medians = [np.median(fed), np.median(fed_ft)]
for i, m in enumerate(medians, start=1):
    ax.text(i + 0.08, m + 0.01, f"{m:.3f}", fontsize=9)

# Optional: tidy y-limits if you want less empty space
# ax.set_ylim(0.0, 1.05)

plt.tight_layout()

# Save for report
plt.savefig("lstm64_32_fl_vs_flft_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()