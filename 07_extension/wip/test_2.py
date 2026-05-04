from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import Helper_functions

"""
Pilot clustering script

Base features:
- 24D average daily load profile (TRAIN only), row-wise z-score normalised

Additional engineered features:
- weekday_weekend_diff
- evening_to_morning_ratio
- overnight_to_day_ratio
- peak_hour_sin
- peak_hour_cos

Feature handling:
- 24D profile is row-wise z-score normalised and then kept as-is
- additional engineered features are standard-scaled column-wise across households
- final clustering matrix = [24D row-z profile | scaled engineered features]

Outputs:
1. household_24h_profiles_raw.csv
2. household_cluster_features_pilot_raw.csv
3. household_cluster_features_pilot_scaled.csv
4. kmeans_k_sweep_metrics_pilot.csv
5. kmeans_assignments_pilot_k{k}.csv
6. cluster_centroids_pilot_k{k}.png
"""

DATA_PATH = Path("../data_files/final_locked_100_normalised.parquet")
MAX_MIN_PATH = Path("../data_files/global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("../data_files/local_kwh_scaler.csv")

OUT_RAW_PROFILES = Path("household_24h_profiles_raw.csv")
OUT_PILOT_RAW = Path("household_cluster_features_pilot_raw.csv")
OUT_PILOT_SCALED = Path("household_cluster_features_pilot_scaled.csv")
OUT_METRICS = Path("kmeans_k_sweep_metrics_pilot.csv")

K_VALUES = [2, 3, 4, 5, 6]
PROFILE_COLS = [f"h{h:02d}" for h in range(24)]
EXTRA_COLS = [
    "weekday_weekend_diff",
    "evening_to_morning_ratio",
    "overnight_to_day_ratio",
    "peak_hour_sin",
    "peak_hour_cos",
]

# --------------------------------------------------
# Step 1: Load data
# --------------------------------------------------
df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING)
)

house_ids = sorted(df["LCLid"].unique())

# --------------------------------------------------
# Step 2: Build raw 24h profiles + extra features
# --------------------------------------------------
raw_profile_rows = []
pilot_feature_rows = []

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

    # Raw 24-hour average daily profile
    hourly_mean = (
        house_train.groupby("hour")["kwh"]
        .mean()
        .reindex(range(24))
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    profile = hourly_mean.to_numpy(dtype=float)

    # Save raw profile for later plotting / interpretation
    raw_row = {"house_id": house_id}
    for h in range(24):
        raw_row[f"h{h:02d}"] = float(profile[h])
    raw_profile_rows.append(raw_row)

    # ---------- row-wise z-score for the 24D profile ----------
    profile_mean = profile.mean()
    profile_std = profile.std()

    if profile_std == 0:
        profile_rowz = np.zeros_like(profile, dtype=float)
    else:
        profile_rowz = (profile - profile_mean) / profile_std

    # ---------- additional engineered features ----------
    weekday_df = house_train[house_train["weekend"] == 0].copy()
    weekend_df = house_train[house_train["weekend"] == 1].copy()

    weekday_mean = (
        weekday_df.groupby("hour")["kwh"]
        .mean()
        .reindex(range(24))
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    weekend_mean = (
        weekend_df.groupby("hour")["kwh"]
        .mean()
        .reindex(range(24))
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    weekday_weekend_diff = float(
        np.mean(np.abs(
            weekday_mean.to_numpy(dtype=float) - weekend_mean.to_numpy(dtype=float)
        ))
    )

    overnight_mean = float(profile[0:6].mean())      # 00-05
    morning_mean = float(profile[6:12].mean())       # 06-11
    afternoon_mean = float(profile[12:18].mean())    # 12-17
    evening_mean = float(profile[18:24].mean())      # 18-23

    evening_to_morning_ratio = (
        evening_mean / morning_mean if morning_mean != 0 else 0.0
    )

    day_mean = (morning_mean + afternoon_mean + evening_mean) / 3
    overnight_to_day_ratio = (
        overnight_mean / day_mean if day_mean != 0 else 0.0
    )

    peak_hour = int(np.argmax(profile))
    peak_hour_sin = float(np.sin(2 * np.pi * peak_hour / 24))
    peak_hour_cos = float(np.cos(2 * np.pi * peak_hour / 24))

    # ---------- combine into one pilot feature row ----------
    row = {"house_id": house_id}

    for h in range(24):
        row[f"h{h:02d}"] = float(profile_rowz[h])

    row["weekday_weekend_diff"] = weekday_weekend_diff
    row["evening_to_morning_ratio"] = evening_to_morning_ratio
    row["overnight_to_day_ratio"] = overnight_to_day_ratio
    row["peak_hour_sin"] = peak_hour_sin
    row["peak_hour_cos"] = peak_hour_cos

    pilot_feature_rows.append(row)

raw_profiles_df = pd.DataFrame(raw_profile_rows)
pilot_features_df = pd.DataFrame(pilot_feature_rows)

if raw_profiles_df.empty or pilot_features_df.empty:
    raise ValueError("No profile/features were created.")

print("\nRaw profiles shape:")
print(raw_profiles_df.shape)
print(raw_profiles_df.head())

print("\nPilot feature matrix shape:")
print(pilot_features_df.shape)
print(pilot_features_df.head())

raw_profiles_df.to_csv(OUT_RAW_PROFILES, index=False)
pilot_features_df.to_csv(OUT_PILOT_RAW, index=False)
'''
# --------------------------------------------------
# Step 3: Scale only the engineered features
# --------------------------------------------------
X_profile = pilot_features_df[PROFILE_COLS].to_numpy(dtype=float)

scaler = StandardScaler()
X_extra_scaled = scaler.fit_transform(pilot_features_df[EXTRA_COLS])

# Final clustering matrix
X_final = np.hstack([X_profile, X_extra_scaled])

# Save a dataframe version of the final clustering matrix
pilot_features_scaled_df = pilot_features_df[["house_id"]].copy()

# keep row-z profile columns unchanged
for col in PROFILE_COLS:
    pilot_features_scaled_df[col] = pilot_features_df[col]

# save scaled engineered features
for i, col in enumerate(EXTRA_COLS):
    pilot_features_scaled_df[col] = X_extra_scaled[:, i]

pilot_features_scaled_df.to_csv(OUT_PILOT_SCALED, index=False)

print("\nFinal clustering matrix shape:")
print(X_final.shape)
'''

# --------------------------------------------------
# Step 3: Use ONLY engineered features
# --------------------------------------------------
scaler = StandardScaler()
X_final = scaler.fit_transform(pilot_features_df[EXTRA_COLS])

pilot_features_scaled_df = pilot_features_df[["house_id"]].copy()
for i, col in enumerate(EXTRA_COLS):
    pilot_features_scaled_df[col] = X_final[:, i]

pilot_features_scaled_df.to_csv(OUT_PILOT_SCALED, index=False)

print("\nFinal clustering matrix shape:")
print(X_final.shape)
# --------------------------------------------------
# Step 4: KMeans sweep
# --------------------------------------------------
results = []

for k in K_VALUES:
    print(f"\nRunning KMeans for k={k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=69,
        n_init=1000
    )

    labels = kmeans.fit_predict(X_final)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_final, labels)
    dbi = davies_bouldin_score(X_final, labels)
    ch = calinski_harabasz_score(X_final, labels)

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

    assignments = pilot_features_df[["house_id"]].copy()
    assignments["cluster"] = labels
    assignments.to_csv(f"kmeans_assignments_pilot_k{k}.csv", index=False)

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
# Step 5: Plot raw 24h profiles by cluster for interpretation
# --------------------------------------------------
for k in K_VALUES:
    assignments = pd.read_csv(f"kmeans_assignments_pilot_k{k}.csv")
    df_merged = raw_profiles_df.merge(assignments, on="house_id")

    cluster_ids = sorted(df_merged["cluster"].unique())
    n_clusters = len(cluster_ids)

    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5), sharey=True)

    if n_clusters == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_df = df_merged[df_merged["cluster"] == cluster_id]

        # plot each household profile faintly
        for _, profile_row in cluster_df.iterrows():
            ax.plot(
                range(24),
                profile_row[PROFILE_COLS].values,
                alpha=0.18,
                linewidth=1
            )

        # mean profile
        mean_profile = cluster_df[PROFILE_COLS].mean()
        ax.plot(
            range(24),
            mean_profile.values,
            linewidth=3,
            label="Mean profile"
        )

        ax.set_title(f"k={k} | Cluster {cluster_id} (n={len(cluster_df)})")
        ax.set_xlabel("Hour of Day")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Scaled kWh Usage")
    fig.suptitle(f"Raw 24h Profiles by Cluster for k={k} (pilot features)")
    fig.tight_layout()
    fig.savefig(f"cluster_centroids_pilot_k{k}.png", dpi=150)
    plt.close(fig)

print("\nDone.")