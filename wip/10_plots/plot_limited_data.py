import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["Pretrained FL + FT", "Pretrained FL", "SARIMA"]
mean_rmse = [0.4352, 0.4530, 0.4853]

# Colours: top red, middle blue, bottom grey
colors = ["red", "tab:blue", "gray"]

y = np.arange(len(methods))

fig, ax = plt.subplots(figsize=(9.3, 2.7))

bars = ax.barh(y, mean_rmse, height=0.45, color=colors)

ax.set_yticks(y)
ax.set_yticklabels(methods)
ax.set_xlabel("Mean RMSE (kWh)")
ax.set_title("Limited-Data Test: Mean RMSE by Method")

# Keep a bit of room for the labels
ax.set_xlim(0, 0.515)

ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.invert_yaxis()

# Value labels
for bar, value in zip(bars, mean_rmse):
    ax.text(
        value + 0.0025,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center",
        ha="left",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("limited_data_horizontal_bar_rmse_coloured.png", dpi=300, bbox_inches="tight")
plt.show()