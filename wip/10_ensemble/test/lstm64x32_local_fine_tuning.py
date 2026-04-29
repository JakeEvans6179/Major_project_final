from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import random
import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

"""
Unseen-house blind test for FL + fine-tuning.

Setup:
- 5 days train
- 2 days val
- 7 days test

Workflow:
1. Load pretrained global FL model
2. For each unseen house:
   - evaluate raw federated model on TEST
   - fine-tune on TRAIN with early stopping on VAL
   - evaluate fine-tuned model on TEST
   - save validation and test predictions for ensemble
3. Save per-house and summary results

Important:
Validation/test windows are built using the immediately preceding split
as context so that forecast origins align with SARIMA.
"""

data_path = Path("unseen_2week_normalised.parquet")
max_min_path = Path("unseen_global_weather_scaler.csv")
local_kwh_scaling = Path("unseen_local_kwh_scaler.csv")

global_model_path = Path("global_chunk_094_LSTM64x32.keras")

HORIZON = 6
WINDOW_SIZE = 24
TARGET_COL = "kwh"

feature_cols = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
]

pred_dir = Path("blind_flft_preds")
pred_dir.mkdir(exist_ok=True)


def compile_for_finetuning(model):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model


def train_model(X_train, y_train, X_val, y_val, global_model):
    model = compile_for_finetuning(global_model)

    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=min(256, len(X_train)),
        verbose=1,
        callbacks=[es]
    )

    return model, history


def make_eval_xy_with_history(history_df, eval_df, window_size, target_col, horizon):
    """
    Build evaluation windows for eval_df using the last `window_size` rows
    of history_df as context.

    This makes the first origin in eval_df use the immediately preceding
    observations from the previous split.
    """
    history_tail = history_df.tail(window_size).copy()
    combined = pd.concat([history_tail, eval_df.copy()], ignore_index=True)

    X_all, y_all = Helper_functions.make_xy(
        combined,
        window_size=window_size,
        target_col=target_col,
        horizon=horizon
    )

    n_eval_origins = len(eval_df) - horizon + 1
    if n_eval_origins <= 0:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0, horizon))

    X_eval = X_all[:n_eval_origins]
    y_eval = y_all[:n_eval_origins]

    return X_eval, y_eval


df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)
)

house_ids = sorted(df["LCLid"].unique())

results = []

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(69)
    np.random.seed(69)
    random.seed(69)

    starting_model = load_model(global_model_path, compile=False)

    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    train_df, val_df, test_df = Helper_functions.get_house_split(df, house_id, feature_cols)

    # Train windows come from train split only
    house_x_train, house_y_train = Helper_functions.make_xy(
        train_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon=HORIZON
    )

    # Validation uses last 24 hours of train as context
    house_x_val, house_y_val = make_eval_xy_with_history(
        history_df=train_df,
        eval_df=val_df,
        window_size=WINDOW_SIZE,
        target_col=TARGET_COL,
        horizon=HORIZON
    )

    # Test uses last 24 hours of val as context
    house_x_test, house_y_test = make_eval_xy_with_history(
        history_df=val_df,
        eval_df=test_df,
        window_size=WINDOW_SIZE,
        target_col=TARGET_COL,
        horizon=HORIZON
    )

    if len(house_x_train) == 0 or len(house_x_val) == 0 or len(house_x_test) == 0:
        print(f"Skipping {house_id}: insufficient samples after windowing")
        continue

    # --------------------------------------------------
    # Raw federated model evaluated on TEST
    # --------------------------------------------------
    pred_scaled_federated = starting_model.predict(house_x_test, verbose=0)

    federated_metrics, y_test_raw, pred_fed_raw = Helper_functions.evaluate_predictions_multistep(
        y_scaled=house_y_test,
        pred_scaled=pred_scaled_federated,
        min_val=kwh_min,
        max_val=kwh_max
    )

    house_test_std = float(np.std(y_test_raw.reshape(-1)))
    if house_test_std == 0:
        house_test_std = np.nan

    # --------------------------------------------------
    # Fine-tune on TRAIN, early stop on VAL
    # --------------------------------------------------
    fine_tuned_model, history = train_model(
        house_x_train, house_y_train,
        house_x_val, house_y_val,
        starting_model
    )

    # --------------------------------------------------
    # Fine-tuned model evaluated on VAL for ensemble alpha search
    # --------------------------------------------------
    pred_scaled_fine_tuned_val = fine_tuned_model.predict(house_x_val, verbose=0)

    fine_tuned_val_metrics, y_ft_val_raw, pred_ft_val_raw = Helper_functions.evaluate_predictions_multistep(
        y_scaled=house_y_val,
        pred_scaled=pred_scaled_fine_tuned_val,
        min_val=kwh_min,
        max_val=kwh_max
    )

    np.savez_compressed(
        pred_dir / f"{house_id}_flft_val.npz",
        y_true_raw=y_ft_val_raw,
        pred_raw=pred_ft_val_raw,
    )

    # --------------------------------------------------
    # Fine-tuned model evaluated on TEST
    # --------------------------------------------------
    pred_scaled_fine_tuned_test = fine_tuned_model.predict(house_x_test, verbose=0)

    fine_tuned_metrics, y_ft_raw, pred_ft_raw = Helper_functions.evaluate_predictions_multistep(
        y_scaled=house_y_test,
        pred_scaled=pred_scaled_fine_tuned_test,
        min_val=kwh_min,
        max_val=kwh_max
    )

    np.savez_compressed(
        pred_dir / f"{house_id}_flft_test.npz",
        y_true_raw=y_ft_raw,
        pred_raw=pred_ft_raw,
    )

    delta_rmse = (
        federated_metrics["mean_rmse_across_horizons"]
        - fine_tuned_metrics["mean_rmse_across_horizons"]
    )
    delta_mae = (
        federated_metrics["mean_mae_across_horizons"]
        - fine_tuned_metrics["mean_mae_across_horizons"]
    )

    federated_nrmse_std = (
        federated_metrics["mean_rmse_across_horizons"] / house_test_std
        if not np.isnan(house_test_std) else np.nan
    )
    fine_tuned_nrmse_std = (
        fine_tuned_metrics["mean_rmse_across_horizons"] / house_test_std
        if not np.isnan(house_test_std) else np.nan
    )
    delta_nrmse_std = (
        federated_nrmse_std - fine_tuned_nrmse_std
        if not np.isnan(federated_nrmse_std) and not np.isnan(fine_tuned_nrmse_std)
        else np.nan
    )

    results.append({
        "house_id": house_id,

        "federated_rmse_t+1": federated_metrics["rmse_t+1"],
        "federated_rmse_t+2": federated_metrics["rmse_t+2"],
        "federated_rmse_t+3": federated_metrics["rmse_t+3"],
        "federated_rmse_t+4": federated_metrics["rmse_t+4"],
        "federated_rmse_t+5": federated_metrics["rmse_t+5"],
        "federated_rmse_t+6": federated_metrics["rmse_t+6"],
        "federated_mae_t+1": federated_metrics["mae_t+1"],
        "federated_mae_t+2": federated_metrics["mae_t+2"],
        "federated_mae_t+3": federated_metrics["mae_t+3"],
        "federated_mae_t+4": federated_metrics["mae_t+4"],
        "federated_mae_t+5": federated_metrics["mae_t+5"],
        "federated_mae_t+6": federated_metrics["mae_t+6"],
        "federated_mean_rmse": federated_metrics["mean_rmse_across_horizons"],
        "federated_mean_mae": federated_metrics["mean_mae_across_horizons"],

        "fine_tuned_rmse_t+1": fine_tuned_metrics["rmse_t+1"],
        "fine_tuned_rmse_t+2": fine_tuned_metrics["rmse_t+2"],
        "fine_tuned_rmse_t+3": fine_tuned_metrics["rmse_t+3"],
        "fine_tuned_rmse_t+4": fine_tuned_metrics["rmse_t+4"],
        "fine_tuned_rmse_t+5": fine_tuned_metrics["rmse_t+5"],
        "fine_tuned_rmse_t+6": fine_tuned_metrics["rmse_t+6"],
        "fine_tuned_mae_t+1": fine_tuned_metrics["mae_t+1"],
        "fine_tuned_mae_t+2": fine_tuned_metrics["mae_t+2"],
        "fine_tuned_mae_t+3": fine_tuned_metrics["mae_t+3"],
        "fine_tuned_mae_t+4": fine_tuned_metrics["mae_t+4"],
        "fine_tuned_mae_t+5": fine_tuned_metrics["mae_t+5"],
        "fine_tuned_mae_t+6": fine_tuned_metrics["mae_t+6"],
        "fine_tuned_mean_rmse": fine_tuned_metrics["mean_rmse_across_horizons"],
        "fine_tuned_mean_mae": fine_tuned_metrics["mean_mae_across_horizons"],

        "fine_tuned_val_mean_rmse": fine_tuned_val_metrics["mean_rmse_across_horizons"],
        "fine_tuned_val_mean_mae": fine_tuned_val_metrics["mean_mae_across_horizons"],

        "delta_rmse": delta_rmse,
        "delta_mae": delta_mae,

        "test_std_kwh": house_test_std,
        "federated_nrmse_std": federated_nrmse_std,
        "fine_tuned_nrmse_std": fine_tuned_nrmse_std,
        "delta_nrmse_std": delta_nrmse_std,

        "epochs_run": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "n_train_windows": len(house_x_train),
        "n_val_windows": len(house_x_val),
        "n_test_windows": len(house_x_test),
    })

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No unseen houses were evaluated.")

results_df.to_csv("FLFT_unseen_2week_per_house_results.csv", index=False)

print("Mean RMSE across horizons federated:", results_df["federated_mean_rmse"].mean())
print("Median RMSE across horizons federated:", results_df["federated_mean_rmse"].median())

print("Mean MAE across horizons federated:", results_df["federated_mean_mae"].mean())
print("Median MAE across horizons federated:", results_df["federated_mean_mae"].median())

print("Mean RMSE across horizons fine tuned:", results_df["fine_tuned_mean_rmse"].mean())
print("Median RMSE across horizons fine tuned:", results_df["fine_tuned_mean_rmse"].median())

print("Mean MAE across horizons fine tuned:", results_df["fine_tuned_mean_mae"].mean())
print("Median MAE across horizons fine tuned:", results_df["fine_tuned_mean_mae"].median())

print("Mean delta RMSE across horizons:", results_df["delta_rmse"].mean())
print("Median delta RMSE across horizons:", results_df["delta_rmse"].median())

print("Mean delta MAE across horizons:", results_df["delta_mae"].mean())
print("Median delta MAE across horizons:", results_df["delta_mae"].median())

print("Mean federated NRMSE_std:", results_df["federated_nrmse_std"].mean())
print("Mean fine-tuned NRMSE_std:", results_df["fine_tuned_nrmse_std"].mean())
print("Mean delta NRMSE_std:", results_df["delta_nrmse_std"].mean())

print("Houses improved in RMSE:", (results_df["delta_rmse"] > 0).sum())
print("Houses worsened in RMSE:", (results_df["delta_rmse"] < 0).sum())

print("Houses improved in MAE:", (results_df["delta_mae"] > 0).sum())
print("Houses worsened in MAE:", (results_df["delta_mae"] < 0).sum())

print("Houses improved in NRMSE_std:", (results_df["delta_nrmse_std"] > 0).sum())
print("Houses worsened in NRMSE_std:", (results_df["delta_nrmse_std"] < 0).sum())

summary_df = pd.DataFrame([{
    "model": "FLFT_unseen_2week_LSTM64x32",
    "checkpoint_id": str(global_model_path),

    "mean_rmse_federated": results_df["federated_mean_rmse"].mean(),
    "median_rmse_federated": results_df["federated_mean_rmse"].median(),
    "mean_mae_federated": results_df["federated_mean_mae"].mean(),
    "median_mae_federated": results_df["federated_mean_mae"].median(),

    "mean_rmse_fine_tuned": results_df["fine_tuned_mean_rmse"].mean(),
    "median_rmse_fine_tuned": results_df["fine_tuned_mean_rmse"].median(),
    "mean_mae_fine_tuned": results_df["fine_tuned_mean_mae"].mean(),
    "median_mae_fine_tuned": results_df["fine_tuned_mean_mae"].median(),

    "mean_delta_rmse": results_df["delta_rmse"].mean(),
    "median_delta_rmse": results_df["delta_rmse"].median(),
    "mean_delta_mae": results_df["delta_mae"].mean(),
    "median_delta_mae": results_df["delta_mae"].median(),

    "mean_federated_nrmse_std": results_df["federated_nrmse_std"].mean(),
    "mean_fine_tuned_nrmse_std": results_df["fine_tuned_nrmse_std"].mean(),
    "mean_delta_nrmse_std": results_df["delta_nrmse_std"].mean(),

    "mean_epochs_run": results_df["epochs_run"].mean(),
    "median_best_epoch": results_df["best_epoch"].median(),
    "n_houses": results_df["house_id"].nunique()
}])

summary_df.to_csv("FLFT_unseen_2week_summary.csv", index=False)
print(summary_df)