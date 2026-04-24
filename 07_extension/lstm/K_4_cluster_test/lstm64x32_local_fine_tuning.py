from pathlib import Path
import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

"""
LSTM64x32 clustered federated fine-tuning on validation set

For each house:
- load its assigned cluster
- load the corresponding clustered FL checkpoint
- evaluate the clustered FL model on validation
- fine-tune locally on train
- evaluate again on validation
"""

# =========================================================
# PATHS
# =========================================================
DATA_PATH = Path("../data_files/final_locked_100_normalised.parquet")
MAX_MIN_PATH = Path("../data_files/global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("../data_files/local_kwh_scaler.csv")

ASSIGNMENT_FILE = Path("kmeans_assignments_rowu_k4.csv")

# ---- choose ONE checkpoint per cluster ----
# Option A: same best overall chunk index for both clusters
CLUSTER_MODEL_PATHS = {
    0: Path("chunk_checkpoints_cluster_0/chunk_038_LSTM64x32_cluster_0.keras"),
    1: Path("chunk_checkpoints_cluster_1/chunk_038_LSTM64x32_cluster_1.keras"),
    2: Path("chunk_checkpoints_cluster_2/chunk_038_LSTM64x32_cluster_2.keras"),
    3: Path("chunk_checkpoints_cluster_3/chunk_038_LSTM64x32_cluster_3.keras"),
}

# If later you want best-per-cluster instead, just change the paths above.

# =========================================================
# SETTINGS
# =========================================================
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

FT_LR = 1e-4
FT_EPOCHS = 100
FT_PATIENCE = 10
BATCH_SIZE = 256

# =========================================================
# HELPERS
# =========================================================
def compile_for_finetuning(model):
    model.compile(
        optimizer=Adam(learning_rate=FT_LR),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model


def train_model(X_train, y_train, X_val, y_val, starting_model):
    model = compile_for_finetuning(starting_model)

    es = EarlyStopping(
        monitor="val_loss",
        patience=FT_PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=FT_EPOCHS,
        batch_size=min(BATCH_SIZE, len(X_train)),
        verbose=1,
        callbacks=[es]
    )

    return model, history


# =========================================================
# LOAD DATA
# =========================================================
df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING)
)

house_ids = sorted(df["LCLid"].astype(str).unique())

# =========================================================
# LOAD CLUSTER ASSIGNMENTS
# =========================================================
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
        f"{len(missing_assignments)} houses missing cluster assignments, e.g. {missing_assignments[:5]}"
    )

# check model files exist
for cluster_id, model_path in CLUSTER_MODEL_PATHS.items():
    if not model_path.exists():
        raise FileNotFoundError(f"Cluster model not found: {model_path}")

print("Cluster model paths:")
for cluster_id, model_path in CLUSTER_MODEL_PATHS.items():
    print(f"  Cluster {cluster_id}: {model_path}")

# =========================================================
# MAIN LOOP
# =========================================================
results = []

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(69)
    np.random.seed(69)
    random.seed(69)

    cluster_id = int(house_to_cluster[house_id])
    model_path = CLUSTER_MODEL_PATHS[cluster_id]

    starting_model = load_model(model_path, compile=False)

    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    train_df, val_df, test_df = Helper_functions.get_house_split(df, house_id, FEATURE_COLS)

    house_x_train, house_y_train = Helper_functions.make_xy(
        train_df,
        window_size=WINDOW_SIZE,
        target_col=TARGET_COL,
        horizon=HORIZON
    )
    house_x_val, house_y_val = Helper_functions.make_xy(
        val_df,
        window_size=WINDOW_SIZE,
        target_col=TARGET_COL,
        horizon=HORIZON
    )

    if len(house_x_train) == 0 or len(house_x_val) == 0:
        print(f"Skipping {house_id}: insufficient samples after windowing")
        continue

    # -------------------------------
    # clustered FL baseline inference
    # -------------------------------
    pred_scaled_clustered = starting_model.predict(house_x_val, verbose=0)

    clustered_metrics, _, _ = Helper_functions.evaluate_predictions_multistep(
        y_scaled=house_y_val,
        pred_scaled=pred_scaled_clustered,
        min_val=kwh_min,
        max_val=kwh_max
    )

    # -------------------------------
    # local fine-tuning
    # -------------------------------
    fine_tuned_model, history = train_model(
        house_x_train, house_y_train,
        house_x_val, house_y_val,
        starting_model
    )

    pred_scaled_fine_tuned = fine_tuned_model.predict(house_x_val, verbose=0)

    fine_tuned_metrics, _, _ = Helper_functions.evaluate_predictions_multistep(
        y_scaled=house_y_val,
        pred_scaled=pred_scaled_fine_tuned,
        min_val=kwh_min,
        max_val=kwh_max
    )

    delta_rmse = (
        clustered_metrics["mean_rmse_across_horizons"]
        - fine_tuned_metrics["mean_rmse_across_horizons"]
    )
    delta_mae = (
        clustered_metrics["mean_mae_across_horizons"]
        - fine_tuned_metrics["mean_mae_across_horizons"]
    )

    results.append({
        "house_id": house_id,
        "cluster": cluster_id,
        "starting_checkpoint": str(model_path),
        "clustered_mean_rmse": clustered_metrics["mean_rmse_across_horizons"],
        "clustered_mean_mae": clustered_metrics["mean_mae_across_horizons"],
        "fine_tuned_mean_rmse": fine_tuned_metrics["mean_rmse_across_horizons"],
        "fine_tuned_mean_mae": fine_tuned_metrics["mean_mae_across_horizons"],
        "delta_rmse": delta_rmse,
        "delta_mae": delta_mae,
        "epochs_run": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1)
    })

# =========================================================
# SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(results)
results_df.to_csv("fine_tuned_LSTM64x32_clustered_per_house.csv", index=False)

print("Mean RMSE across horizons clustered FL:", results_df["clustered_mean_rmse"].mean())
print("Median RMSE across horizons clustered FL:", results_df["clustered_mean_rmse"].median())

print("Mean MAE across horizons clustered FL:", results_df["clustered_mean_mae"].mean())
print("Median MAE across horizons clustered FL:", results_df["clustered_mean_mae"].median())

print("Mean RMSE across horizons clustered FL + fine-tuning:", results_df["fine_tuned_mean_rmse"].mean())
print("Median RMSE across horizons clustered FL + fine-tuning:", results_df["fine_tuned_mean_rmse"].median())

print("Mean MAE across horizons clustered FL + fine-tuning:", results_df["fine_tuned_mean_mae"].mean())
print("Median MAE across horizons clustered FL + fine-tuning:", results_df["fine_tuned_mean_mae"].median())

print("Mean delta RMSE across horizons:", results_df["delta_rmse"].mean())
print("Median delta RMSE across horizons:", results_df["delta_rmse"].median())

print("Mean delta MAE across horizons:", results_df["delta_mae"].mean())
print("Median delta MAE across horizons:", results_df["delta_mae"].median())

print("Houses improved in RMSE:", (results_df["delta_rmse"] > 0).sum())
print("Houses worsened in RMSE:", (results_df["delta_rmse"] < 0).sum())

print("Houses improved in MAE:", (results_df["delta_mae"] > 0).sum())
print("Houses worsened in MAE:", (results_df["delta_mae"] < 0).sum())

# optional per-cluster summaries
cluster_summary_df = (
    results_df.groupby("cluster")
    .agg(
        n_houses=("house_id", "count"),
        mean_clustered_rmse=("clustered_mean_rmse", "mean"),
        mean_fine_tuned_rmse=("fine_tuned_mean_rmse", "mean"),
        mean_clustered_mae=("clustered_mean_mae", "mean"),
        mean_fine_tuned_mae=("fine_tuned_mean_mae", "mean"),
        mean_delta_rmse=("delta_rmse", "mean"),
        mean_delta_mae=("delta_mae", "mean"),
        mean_epochs_run=("epochs_run", "mean"),
        mean_best_epoch=("best_epoch", "mean"),
    )
    .reset_index()
)
cluster_summary_df.to_csv("fine_tuned_LSTM64x32_clustered_summary_by_cluster.csv", index=False)

summary_df = pd.DataFrame([{
    "model": "fine_tuned_LSTM64x32_clustered",
    "cluster_model_paths": str(CLUSTER_MODEL_PATHS),
    "Mean RMSE across horizons clustered FL": results_df["clustered_mean_rmse"].mean(),
    "Median RMSE across horizons clustered FL": results_df["clustered_mean_rmse"].median(),
    "Mean MAE across horizons clustered FL": results_df["clustered_mean_mae"].mean(),
    "Median MAE across horizons clustered FL": results_df["clustered_mean_mae"].median(),
    "Mean RMSE across horizons clustered FL + fine-tuning": results_df["fine_tuned_mean_rmse"].mean(),
    "Median RMSE across horizons clustered FL + fine-tuning": results_df["fine_tuned_mean_rmse"].median(),
    "Mean MAE across horizons clustered FL + fine-tuning": results_df["fine_tuned_mean_mae"].mean(),
    "Median MAE across horizons clustered FL + fine-tuning": results_df["fine_tuned_mean_mae"].median(),
    "Mean delta RMSE across horizons": results_df["delta_rmse"].mean(),
    "Median delta RMSE across horizons": results_df["delta_rmse"].median(),
    "Mean delta MAE across horizons": results_df["delta_mae"].mean(),
    "Median delta MAE across horizons": results_df["delta_mae"].median(),
    "Mean epochs run": results_df["epochs_run"].mean(),
    "Mean best epoch": results_df["best_epoch"].mean()
}])

summary_df.to_csv("fine_tuned_LSTM64x32_clustered_summary.csv", index=False)
print(summary_df)