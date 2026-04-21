from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

"""
LSTM64x32 clustered federated training - validation screening

For each communication chunk:
- load one checkpoint per cluster
- evaluate each household using the model from its assigned cluster
- compute overall mean validation RMSE across households
"""

# =========================
# PATHS
# =========================
DATA_PATH = Path("../data_files/final_locked_100_normalised.parquet")
MAX_MIN_PATH = Path("../data_files/global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("../data_files/local_kwh_scaler.csv")

ASSIGNMENT_FILE = Path("kmeans_assignments_rowz_k4.csv")

# cluster checkpoint folders from your clustered training runs
CLUSTER_CHECKPOINT_DIRS = {
    0: Path("chunk_checkpoints_cluster_0"),
    1: Path("chunk_checkpoints_cluster_1"),
    2: Path("chunk_checkpoints_cluster_2"),
    3: Path("chunk_checkpoints_cluster_3"),
}

NUM_CHUNKS = 40

# =========================
# FORECAST SETTINGS
# =========================
HORIZON = 6
WINDOW_SIZE = 24
TARGET_COL = "kwh"

FEATURE_COLS = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
]

# =========================
# LOAD DATA
# =========================
df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING)
)

house_ids = sorted(df["LCLid"].astype(str).unique())

# =========================
# LOAD CLUSTER ASSIGNMENTS
# =========================
assignments_df = pd.read_csv(ASSIGNMENT_FILE)
assignments_df["house_id"] = assignments_df["house_id"].astype(str)

required_cols = {"house_id", "cluster"}
if not required_cols.issubset(assignments_df.columns):
    raise ValueError(
        f"Assignment file must contain columns {required_cols}, "
        f"but got {assignments_df.columns.tolist()}"
    )

house_to_cluster = dict(zip(assignments_df["house_id"], assignments_df["cluster"]))

missing_assignments = [h for h in house_ids if h not in house_to_cluster]
if missing_assignments:
    raise ValueError(
        f"{len(missing_assignments)} houses are missing cluster assignments, e.g. {missing_assignments[:5]}"
    )

clusters_in_assignments = sorted(assignments_df["cluster"].unique())
print("Clusters found:", clusters_in_assignments)

# =========================
# PREP VALIDATION DATA
# =========================
val_data = {}

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    train_df, val_df, test_df = Helper_functions.get_house_split(df, house_id, FEATURE_COLS)

    house_x_val, house_y_val = Helper_functions.make_xy(
        val_df,
        window_size=WINDOW_SIZE,
        target_col=TARGET_COL,
        horizon=HORIZON
    )

    if len(house_x_val) == 0:
        print(f"Skipping {house_id}: insufficient validation samples after windowing")
        continue

    val_data[house_id] = {
        "x_val": house_x_val,
        "y_val": house_y_val,
        "kwh_min": float(kwh_min),
        "kwh_max": float(kwh_max),
        "cluster": int(house_to_cluster[house_id]),
    }

print("Number of houses in val_data:", len(val_data))

# =========================
# SCREEN CHUNKS
# =========================
chunk_val_metrics = {}
chunk_cluster_metrics = []

for chunk_idx in range(1, NUM_CHUNKS + 1):
    print(f"\n=== Evaluating chunk {chunk_idx}/{NUM_CHUNKS} ===")
    tf.keras.backend.clear_session()

    # load one model per cluster for this chunk
    cluster_models = {}

    for cluster_id in clusters_in_assignments:
        model_path = (
            CLUSTER_CHECKPOINT_DIRS[cluster_id]
            / f"chunk_{chunk_idx:03d}_LSTM64x32_cluster_{cluster_id}.keras"
        )

        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")

        print(f"Loading cluster {cluster_id} model from {model_path}")
        cluster_models[cluster_id] = load_model(model_path)

    overall_house_metrics = []
    per_cluster_house_metrics = {cluster_id: [] for cluster_id in clusters_in_assignments}

    # evaluate each household using its own cluster model
    for house_id, house_info in val_data.items():
        cluster_id = house_info["cluster"]
        model = cluster_models[cluster_id]

        pred_scaled = model.predict(house_info["x_val"], verbose=0)

        metrics, y_raw, pred_raw = Helper_functions.evaluate_predictions_multistep(
            y_scaled=house_info["y_val"],
            pred_scaled=pred_scaled,
            min_val=house_info["kwh_min"],
            max_val=house_info["kwh_max"],
        )

        house_rmse = metrics["mean_rmse_across_horizons"]

        overall_house_metrics.append(house_rmse)
        per_cluster_house_metrics[cluster_id].append(house_rmse)

    # overall mean across households
    mean_val_rmse = float(np.mean(overall_house_metrics))
    chunk_val_metrics[chunk_idx] = mean_val_rmse

    row = {
        "chunk": chunk_idx,
        "overall_mean_rmse_kwh": mean_val_rmse,
    }

    for cluster_id in clusters_in_assignments:
        cluster_rmse = float(np.mean(per_cluster_house_metrics[cluster_id]))
        row[f"cluster_{cluster_id}_mean_rmse_kwh"] = cluster_rmse
        row[f"cluster_{cluster_id}_n_houses"] = len(per_cluster_house_metrics[cluster_id])

    chunk_cluster_metrics.append(row)

    print(f"Chunk {chunk_idx}: Overall mean RMSE across horizons = {mean_val_rmse:.4f}")
    for cluster_id in clusters_in_assignments:
        print(
            f"  Cluster {cluster_id}: "
            f"mean RMSE = {row[f'cluster_{cluster_id}_mean_rmse_kwh']:.4f} "
            f"(n={row[f'cluster_{cluster_id}_n_houses']})"
        )

    # cleanup
    del cluster_models
    import gc
    gc.collect()

# =========================
# SAVE RESULTS
# =========================
summary_df = pd.DataFrame(chunk_cluster_metrics)
summary_df["model"] = "LSTM64x32_clustered_federated"
summary_df.to_csv("chunk_validation_results_clustered.csv", index=False)

best_chunk = int(summary_df.loc[summary_df["overall_mean_rmse_kwh"].idxmin(), "chunk"])
best_rmse = float(summary_df["overall_mean_rmse_kwh"].min())

print("\nBest chunk:", best_chunk)
print("Best overall mean RMSE:", best_rmse)

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 6))
plt.plot(summary_df["chunk"], summary_df["overall_mean_rmse_kwh"], marker="o", label="Overall")

for cluster_id in clusters_in_assignments:
    plt.plot(
        summary_df["chunk"],
        summary_df[f"cluster_{cluster_id}_mean_rmse_kwh"],
        marker="o",
        linestyle="--",
        label=f"Cluster {cluster_id}"
    )

plt.title("Clustered FL: Mean RMSE across horizons vs Communication Chunks")
plt.xlabel("Communication Chunk")
plt.ylabel("Mean RMSE across horizons")
plt.legend()
plt.tight_layout()
plt.savefig("validation_loss_vs_chunks_clustered.png", dpi=200)
plt.close()