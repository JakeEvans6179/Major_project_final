from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Helper_functions

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_PATH = Path("data_files/final_locked_100_normalised.parquet")
MAX_MIN_PATH = Path("data_files/global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("data_files/local_kwh_scaler.csv")

FEATURE_COLS = [f"h{h:02d}" for h in range(24)]
K_VALUES = [2, 3, 4]
PANEL_LABELS = ["(a)", "(b)", "(c)"]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING)
)

house_ids = sorted(df["LCLid"].unique())

# --------------------------------------------------
# BUILD 24-HOUR AVERAGE DAILY PROFILES FROM TRAIN DATA ONLY
# --------------------------------------------------
profile_rows = []

for house_id in house_ids:
    house_train = df[
        (df["LCLid"] == house_id) &
        (df["split"] == "train")
    ].copy()

    if house_train.empty:
        continue

    house_train["hour"] = house_train["DateTime"].dt.hour

    hourly_mean = (
        house_train.groupby("hour")["kwh"]
        .mean()
        .reindex(range(24))
    )

    # Fill any missing hours just in case
    hourly_mean = hourly_mean.interpolate(limit_direction="both").ffill().bfill()

    row = {"house_id": house_id}
    for h in range(24):
        row[f"h{h:02d}"] = float(hourly_mean.loc[h])

    profile_rows.append(row)

hourly_profiles = pd.DataFrame(profile_rows)

# --------------------------------------------------
# ROW-WISE UNIT NORMALISATION
# clustering focuses on shape rather than magnitude
# --------------------------------------------------
X_raw = hourly_profiles[FEATURE_COLS].copy()

row_norms = np.sqrt((X_raw ** 2).sum(axis=1)).replace(0, np.nan)
X = X_raw.div(row_norms, axis=0).fillna(0.0)

# --------------------------------------------------
# PLOT
# --------------------------------------------------
fig, axes = plt.subplots(
    nrows=3, ncols=1,
    figsize=(8.2, 7.8),
    sharex=True,
    sharey=True
)

for ax, k, panel in zip(axes, K_VALUES, PANEL_LABELS):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=100,
        max_iter=1000,
        tol=1e-6
    )

    labels = kmeans.fit_predict(X)

    temp = X.copy()
    temp["cluster"] = labels

    for cluster_id in sorted(temp["cluster"].unique()):
        group = temp[temp["cluster"] == cluster_id]
        mean_profile = group[FEATURE_COLS].mean().values

        ax.plot(
            range(24),
            mean_profile,
            linewidth=2,
            label=f"Cluster {cluster_id} (n={len(group)})"
        )

    ax.set_title(f"{panel} Cluster centroids for k = {k}", loc="left", fontsize=11)
    ax.set_ylabel("Average normalised demand")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper right",
        fontsize=8,
        frameon=False,
        ncol=2
    )

axes[-1].set_xlabel("Hour of day")
axes[-1].set_xticks(range(0, 24, 2))

fig.suptitle("Cluster centroid profiles based on average 24-hour household demand shape", y=0.995, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig("cluster_centroids_k234_stacked.png", dpi=300, bbox_inches="tight")
plt.show()