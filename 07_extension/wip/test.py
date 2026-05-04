from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import Helper_functions

"""
Build 48D household profiles using TRAIN data only:
- 24-hour average WEEKDAY profile
- 24-hour average WEEKEND profile

Then apply row-wise z-score normalisation so clustering focuses on SHAPE
rather than absolute profile level.

Outputs:
1. household_48h_train_profiles_raw.csv
   - one row per house
   - wd_h00 ... wd_h23 = average weekday profile
   - we_h00 ... we_h23 = average weekend profile

2. household_48h_train_profiles_rowz.csv
   - same layout, but each row z-scored across its own 48 values

3. kmeans_k_sweep_metrics_48d_rowz.csv
   - clustering metrics for each k

4. kmeans_assignments_48d_rowz_k{k}.csv
   - house_id to cluster assignment for each k

5. cluster_centroids_48d_rowz_k{k}.png
   - mean raw weekday/weekend profiles per cluster for interpretation
"""

DATA_PATH = Path("../data_files/final_locked_100_normalised.parquet")
MAX_MIN_PATH = Path("../data_files/global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("../data_files/local_kwh_scaler.csv")

OUT_RAW = Path("household_48h_train_profiles_raw.csv")
OUT_ROWZ = Path("household_48h_train_profiles_rowz.csv")
OUT_METRICS = Path("kmeans_k_sweep_metrics_48d_rowz.csv")

K_VALUES = [2, 3, 4, 5, 6]

WEEKDAY_COLS = [f"wd_h{h:02d}" for h in range(24)]
WEEKEND_COLS = [f"we_h{h:02d}" for h in range(24)]
FEATURE_COLS = WEEKDAY_COLS + WEEKEND_COLS

# --------------------------------------------------
# Step 1: Load data
# --------------------------------------------------
df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING)
)

house_ids = sorted(df["LCLid"].unique())

# --------------------------------------------------
# Step 2: Build raw 48D profiles
# --------------------------------------------------
profile_rows = []

for house_id in house_ids:
    print(f"Processing {house_id}")

    house_train = df[
        (df["LCLid"] == house_id) &
        (df["split"] == "train")
    ].copy()

    if house_train.empty:
        print(f"Skipping {house_id}: empty train split")
        continue

    house_train["hour"] = house_train["DateTime"].dt.hour

    # Assuming weekend column is already 0/1 in your dataset
    weekday_df = house_train[house_train["weekend"] == 0].copy()
    weekend_df = house_train[house_train["weekend"] == 1].copy()

    # Average weekday hourly profile
    weekday_mean = (
        weekday_df.groupby("hour")["kwh"]
        .mean()
        .reindex(range(24))
    )

    # Average weekend hourly profile
    weekend_mean = (
        weekend_df.groupby("hour")["kwh"]
        .mean()
        .reindex(range(24))
    )

    # Safety fill in case an hour is missing
    weekday_mean = weekday_mean.interpolate(limit_direction="both").ffill().bfill()
    weekend_mean = weekend_mean.interpolate(limit_direction="both").ffill().bfill()

    row = {"house_id": house_id}

    for h in range(24):
        row[f"wd_h{h:02d}"] = float(weekday_mean.loc[h])

    for h in range(24):
        row[f"we_h{h:02d}"] = float(weekend_mean.loc[h])

    profile_rows.append(row)

hourly_profiles_raw = pd.DataFrame(profile_rows)

if hourly_profiles_raw.empty:
    raise ValueError("No household 48D profiles were created.")

print("\nRaw 48D profile matrix shape:")
print(hourly_profiles_raw.shape)
print(hourly_profiles_raw.head())

hourly_profiles_raw.to_csv(OUT_RAW, index=False)

# --------------------------------------------------
# Step 3: Row-wise z-score normalisation
# --------------------------------------------------
cluster_features_raw = hourly_profiles_raw[FEATURE_COLS].copy()

row_means = cluster_features_raw.mean(axis=1)
row_stds = cluster_features_raw.std(axis=1).replace(0, np.nan)

cluster_features_rowz = (
    cluster_features_raw
    .sub(row_means, axis=0)
    .div(row_stds, axis=0)
    .fillna(0.0)
)

hourly_profiles_rowz = hourly_profiles_raw[["house_id"]].copy()
for col in FEATURE_COLS:
    hourly_profiles_rowz[col] = cluster_features_rowz[col]

print("\nRow-z 48D profile matrix shape:")
print(hourly_profiles_rowz.shape)
print(hourly_profiles_rowz.head())

hourly_profiles_rowz.to_csv(OUT_ROWZ, index=False)

# This is what KMeans will use
X = cluster_features_rowz.copy()

# --------------------------------------------------
# Step 4: KMeans sweep
# --------------------------------------------------
results = []

for k in K_VALUES:
    print(f"\nRunning KMeans for k={k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=50
    )

    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    min_cluster_size = int(cluster_sizes.min())
    max_cluster_size = int(cluster_sizes.max())

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette,
        "davies_bouldin": dbi,
        "calinski_harabasz": ch,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "cluster_sizes": str(cluster_sizes.to_dict())
    })

    assignments = hourly_profiles_raw[["house_id"]].copy()
    assignments["cluster"] = labels
    assignments.to_csv(f"kmeans_assignments_48d_rowz_k{k}.csv", index=False)

    print(f"k={k} cluster sizes: {cluster_sizes.to_dict()}")
    print(f"inertia={inertia:.4f}")
    print(f"silhouette={silhouette:.4f}")
    print(f"DBI={dbi:.4f}")
    print(f"CH={ch:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_METRICS, index=False)

print("\nKMeans sweep results:")
print(results_df)

# --------------------------------------------------
# Step 5: Plot cluster mean RAW profiles for interpretation
# --------------------------------------------------
for k in K_VALUES:
    assignments = pd.read_csv(f"kmeans_assignments_48d_rowz_k{k}.csv")
    df_merged = hourly_profiles_raw.merge(assignments, on="house_id")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Weekday panel
    for cluster_id, group in df_merged.groupby("cluster", sort=True):
        mean_weekday = group[WEEKDAY_COLS].mean().to_numpy()
        axes[0].plot(
            range(24),
            mean_weekday,
            label=f"Cluster {cluster_id} (n={len(group)})",
            lw=2
        )

    axes[0].set_title(f"Weekday Profiles for k={k}")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Scaled kWh Usage")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Weekend panel
    for cluster_id, group in df_merged.groupby("cluster", sort=True):
        mean_weekend = group[WEEKEND_COLS].mean().to_numpy()
        axes[1].plot(
            range(24),
            mean_weekend,
            label=f"Cluster {cluster_id} (n={len(group)})",
            lw=2
        )

    axes[1].set_title(f"Weekend Profiles for k={k}")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Average 48D Profiles for k={k} (clustered on row-z features)")
    fig.tight_layout()
    fig.savefig(f"cluster_centroids_48d_rowz_k{k}.png", dpi=150)
    plt.close(fig)

print("\nDone.")