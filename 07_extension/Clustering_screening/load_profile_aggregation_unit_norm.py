from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import Helper_functions

"""
Build 24-hour average daily load profiles for each house using training data only,
apply row-wise unit normalisation so clustering focuses on shape 
Run Kmeans for k=2 to 6 and save metrics for each K and cluster assignments for each house.
"""

DATA_PATH = Path("../data_files/final_locked_100_normalised.parquet")
MAX_MIN_PATH = Path("../data_files/global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("../data_files/local_kwh_scaler.csv")

OUT_RAW = Path("household_24h_train_profiles_raw.csv")
OUT_ROWU = Path("household_24h_train_profiles_rowu.csv")
OUT_METRICS = Path("kmeans_k_sweep_metrics_rowu.csv")

K_VALUES = [2, 3, 4, 5, 6]
FEATURE_COLS = [f"h{h:02d}" for h in range(24)]

#load data
df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING)
)

house_ids = sorted(df["LCLid"].unique())

#build average 24-hour profiles for each house using training data only
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

    hourly_mean = (
        house_train.groupby("hour")["kwh"]
        .mean()
        .reindex(range(24))
    )

    #ffill, bfil just in case
    hourly_mean = hourly_mean.interpolate(limit_direction="both").ffill().bfill()

    row = {"house_id": house_id}
    for h in range(24):
        row[f"h{h:02d}"] = float(hourly_mean.loc[h])

    profile_rows.append(row)

hourly_profiles_raw = pd.DataFrame(profile_rows)

if hourly_profiles_raw.empty:
    raise ValueError("No household profiles were created.")

print("\nRaw profile matrix shape:")
print(hourly_profiles_raw.shape)
print(hourly_profiles_raw.head())

hourly_profiles_raw.to_csv(OUT_RAW, index=False)

#unit normalisation per row so clustering focuses on shape rather than magnitude
cluster_features_raw = hourly_profiles_raw[FEATURE_COLS].copy()

row_norms = np.sqrt((cluster_features_raw ** 2).sum(axis=1)).replace(0, np.nan)

cluster_features_rowu = (
    cluster_features_raw
    .div(row_norms, axis=0)
    .fillna(0.0)
)

hourly_profiles_rowu = hourly_profiles_raw[["house_id"]].copy()
for col in FEATURE_COLS:
    hourly_profiles_rowu[col] = cluster_features_rowu[col]

print("\nRow-unit profile matrix shape:")
print(hourly_profiles_rowu.shape)
print(hourly_profiles_rowu.head())

hourly_profiles_rowu.to_csv(OUT_ROWU, index=False)

#prepare KMeans input
X = cluster_features_rowu.copy()

#Kmeans clustering and eval
results = []

for k in K_VALUES:
    print(f"\nRunning KMeans for k={k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=1000,        #perform 1000 random initialisations to avoid local minima
        max_iter=1000,      #perfrom up to 1000 iterations per run to ensure convergence
        tol=1e-6,
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

    # Save assignments
    assignments = hourly_profiles_raw[["house_id"]].copy()
    assignments["cluster"] = labels
    assignments.to_csv(f"kmeans_assignments_rowu_k{k}.csv", index=False)

    print(f"k={k} cluster sizes: {cluster_sizes.to_dict()}")
    print(f"inertia={inertia:.4f}")
    print(f"silhouette={silhouette:.4f}")
    print(f"DBI={dbi:.4f}")
    print(f"CH={ch:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_METRICS, index=False)

print("\nKMeans sweep results:")
print(results_df)

#plot cluster centroids for each K
for k in K_VALUES:
    assignments = pd.read_csv(f"kmeans_assignments_rowu_k{k}.csv")
    df_merged = hourly_profiles_raw.merge(assignments, on="house_id")

    plt.figure(figsize=(10, 5))

    for cluster_id, group in df_merged.groupby("cluster", sort=True):
        mean_profile = group[FEATURE_COLS].mean()
        plt.plot(
            range(24),
            mean_profile,
            label=f"Cluster {cluster_id} (n={len(group)})",
            lw=2
        )

    plt.title(f"Average 24-Hour Profiles for k={k} (clustered on row-unit features)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Scaled kWh Usage")
    plt.xticks(range(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cluster_centroids_rowu_k{k}.png", dpi=150)
    plt.close()

print("\nDone.")