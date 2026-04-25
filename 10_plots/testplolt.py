import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# DATA
# =========================
data = {
    "Centralised": {
        "CNN-LSTM": 0.3441,
        "CNN-LSTM-Dense": 0.3439,
        "LSTM64_Dense": 0.3453,
        "LSTM64_32": 0.3438,
    },
    "Centralised + FT": {
        "CNN-LSTM": 0.3255,
        "CNN-LSTM-Dense": 0.3261,
        "LSTM64_Dense": 0.3260,
        "LSTM64_32": 0.3266,
    },
    "Localised": {
        "CNN-LSTM": 0.3393,
        "CNN-LSTM-Dense": 0.3406,
        "LSTM64_Dense": 0.3412,
        "LSTM64_32": 0.3399,
    },
    "Federated": {
        "CNN-LSTM": 0.3892,
        "CNN-LSTM-Dense": 0.3860,
        "LSTM64_Dense": 0.3911,
        "LSTM64_32": 0.3832,
    },
    "Federated + FT": {
        "CNN-LSTM": 0.3308,
        "CNN-LSTM-Dense": 0.3305,
        "LSTM64_Dense": 0.3298,
        "LSTM64_32": 0.3276,
    }
}

df = pd.DataFrame(data)

df = df[[
    "Centralised",
    "Centralised + FT",
    "Localised",
    "Federated",
    "Federated + FT"
]]

df = df.loc[[
    "CNN-LSTM",
    "CNN-LSTM-Dense",
    "LSTM64_Dense",
    "LSTM64_32"
]]

# =========================
# FIXED COLOUR MAP
# =========================
model_colors = {
    "CNN-LSTM": "blue",
    "CNN-LSTM-Dense": "orange",
    "LSTM64_32": "red",
    "LSTM64_Dense": "green",
}

# =========================
# PLOT
# =========================
x = np.arange(len(df.columns))

fig, ax = plt.subplots(figsize=(11, 4))

for model in df.index:
    ax.plot(
        x,
        df.loc[model].values,
        marker="o",
        linewidth=2,
        markersize=6,
        label=model,
        color=model_colors[model]
    )

ax.set_xticks(x)
ax.set_xticklabels(df.columns, rotation=20, ha="right")
ax.set_ylabel("Mean RMSE (kWh)")
ax.set_xlabel("Training Regime")
ax.set_title("Performance Across Training Regimes")

ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig("summary_comparison_lineplot.png", dpi=300, bbox_inches="tight")
plt.close()
print("done")