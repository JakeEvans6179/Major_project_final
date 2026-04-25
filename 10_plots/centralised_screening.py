import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# -----------------------------
# Centralised screening results
# -----------------------------
data = [
    ("LSTM64_32", 0.3438, 0.2190),
    ("CNN-LSTM-Dense", 0.3439, 0.2211),
    ("CNN-LSTM", 0.3441, 0.2192),
    ("CNN-2LSTM-Dense", 0.3446, 0.2198),
    ("2CNN-LSTM", 0.3449, 0.2151),
    ("LSTM64_32-Dense", 0.3451, 0.2194),
    ("LSTM64-Dense", 0.3453, 0.2198),
    ("LSTM64", 0.3453, 0.2208),
    ("Seq2Seq", 0.3458, 0.2244),
    ("LSTM20_20", 0.3461, 0.2231),
    ("2CNN-LSTM-Dense", 0.3463, 0.2208),
    #("Dense", 0.3606, 0.2403),
]

df = pd.DataFrame(data, columns=["Model", "Mean_RMSE", "Mean_MAE"])

shortlisted = {"LSTM64_32", "CNN-LSTM-Dense", "CNN-LSTM", "LSTM64-Dense"}

# Sort best to worst
df = df.sort_values("Mean_RMSE", ascending=True).reset_index(drop=True)

# Y positions
y = range(len(df))

# Colour by shortlist status
colors = ["tab:blue" if m in shortlisted else "lightgray" for m in df["Model"]]

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# horizontal stems
for yi, xi, c in zip(y, df["Mean_RMSE"], colors):
    ax.hlines(y=yi, xmin=df["Mean_RMSE"].min() - 0.0003, xmax=xi, color=c, linewidth=2)

# dots
ax.scatter(df["Mean_RMSE"], y, s=90, c=colors, edgecolors="black", zorder=3)

# value labels
for yi, xi in zip(y, df["Mean_RMSE"]):
    ax.text(xi + 0.00015, yi, f"{xi:.4f}", va="center", fontsize=8)

# axes / labels
ax.set_yticks(list(y))
ax.set_yticklabels(df["Model"])
ax.invert_yaxis()  # best at top

ax.set_xlabel("Mean RMSE")
ax.set_title("Centralised Architecture Screening")

# x-limits: leave margin, but keep Dense visible
xmin = df["Mean_RMSE"].min() - 0.0005
xmax = df["Mean_RMSE"].max() + 0.0010
ax.set_xlim(xmin, xmax)

ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Shortlisted',
           markerfacecolor='tab:blue', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Not shortlisted',
           markerfacecolor='lightgray', markeredgecolor='black', markersize=8),
]
ax.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.savefig("centralised_screening_dotplot.png", dpi=300, bbox_inches="tight")
plt.close()