from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import random
import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

'''
Testing for LSTM64x32
6 hour forecasting horizon 

validation metrics
'''

plots_dir = Path("house_plots_lstm64x32")
plots_dir.mkdir(exist_ok=True) # This creates the folder if it doesn't exist
'''
tf.random.set_seed(69)
np.random.seed(69)       # Ensures NumPy windowing/math is consistent
random.seed(69)          # Ensures Python's internal loops are consistent
'''

data_path = Path("../03_feature_engineering/final_locked_100_normalised.parquet")

max_min_path = Path("../03_feature_engineering/global_weather_scaler.csv")

local_kwh_scaling = Path("../03_feature_engineering/local_kwh_scaler.csv")


HORIZON = 6
WINDOW_SIZE = 24
TARGET_COL = "kwh"


#model input features
feature_cols = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
]



#build model and compiler
def build_nn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences = True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(HORIZON, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model



def train_one_model(X_train, y_train, X_val, y_val):
    model = build_nn(X_train.shape[1:])


    es = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=256,
        verbose=0,
        callbacks=[es]
    )

    return model, history




df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)   #load global weather scalers and local kwh scalers
#house_ids = sorted(df["LCLid"].unique())[:15]
house_ids = sorted(df["LCLid"].unique())
results = []

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    train_df, val_df, _ = Helper_functions.get_house_split(df, house_id, feature_cols)

    X_train, y_train = Helper_functions.make_xy(train_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon =HORIZON) #output is a vector of 6 values now
    X_val, y_val = Helper_functions.make_xy(val_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon =HORIZON)
    

    if len(X_train) == 0 or len(X_val) == 0:
        print(f"Skipping {house_id}: insufficient samples after windowing")
        continue

    tf.keras.backend.clear_session()
    tf.random.set_seed(69)
    np.random.seed(69)
    random.seed(69)

    model, history = train_one_model(X_train, y_train, X_val, y_val)
    pred_scaled = model.predict(X_val, verbose=0)


    metrics, y_raw, pred_raw = Helper_functions.evaluate_predictions_multistep(
        y_scaled=y_val,
        pred_scaled=pred_scaled,
        min_val=kwh_min,
        max_val=kwh_max
    )

    #check to see if there is overfitting
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    epochs_run = len(history.history["loss"])

    train_loss_at_best_val = float(history.history["loss"][best_epoch - 1])
    best_val_loss = float(history.history["val_loss"][best_epoch - 1])
    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    generalisation_gap = best_val_loss - train_loss_at_best_val

    print("\nMulti-step metrics:")
    print("RMSE t+1:", metrics["rmse_t+1"])
    print("RMSE t+2:", metrics["rmse_t+2"])
    print("RMSE t+3:", metrics["rmse_t+3"])
    print("RMSE t+4:", metrics["rmse_t+4"])
    print("RMSE t+5:", metrics["rmse_t+5"])
    print("RMSE t+6:", metrics["rmse_t+6"])

    print("MAE t+1:", metrics["mae_t+1"])
    print("MAE t+2:", metrics["mae_t+2"])
    print("MAE t+3:", metrics["mae_t+3"])
    print("MAE t+4:", metrics["mae_t+4"])
    print("MAE t+5:", metrics["mae_t+5"])
    print("MAE t+6:", metrics["mae_t+6"])

    print("Mean RMSE across horizons:", metrics["mean_rmse_across_horizons"])
    print("Mean MAE across horizons:", metrics["mean_mae_across_horizons"])

    results.append({
        "house_id": house_id,
        "rmse_t+1": metrics["rmse_t+1"],
        "rmse_t+2": metrics["rmse_t+2"],
        "rmse_t+3": metrics["rmse_t+3"],
        "rmse_t+4": metrics["rmse_t+4"],
        "rmse_t+5": metrics["rmse_t+5"],
        "rmse_t+6": metrics["rmse_t+6"],

        "mae_t+1": metrics["mae_t+1"],
        "mae_t+2": metrics["mae_t+2"],
        "mae_t+3": metrics["mae_t+3"],
        "mae_t+4": metrics["mae_t+4"],
        "mae_t+5": metrics["mae_t+5"],
        "mae_t+6": metrics["mae_t+6"],

        "mean_rmse_across_horizons": metrics["mean_rmse_across_horizons"],
        "mean_mae_across_horizons": metrics["mean_mae_across_horizons"],
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "train_loss_at_best_val": train_loss_at_best_val,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "generalisation_gap": generalisation_gap,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
    })
    
    plot_horizon = min(14 * 24, len(y_raw))

    # ==========================================
    # PLOT 1: Forecast for 1 Hour Ahead (t+1)
    # Using column index 0: y_raw[:, 0]
    # ==========================================
    plt.figure(figsize=(12, 4))
    plt.plot(y_raw[:plot_horizon, 0], label="Actual t+1")
    plt.plot(pred_raw[:plot_horizon, 0], label="Predicted t+1")
    plt.xlabel("Val sample index")
    plt.ylabel("kWh")
    plt.title(f"{house_id} | 6-step forecast | LSTM64x32 | Horizon t+1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir/f"{house_id}_6step_tplus1.png", dpi=200)
    plt.close()

    # ==========================================
    # PLOT 2: Forecast for 3 Hours Ahead (t+3)
    # Using column index 2: y_raw[:, 2]
    # ==========================================
    plt.figure(figsize=(12, 4))
    plt.plot(y_raw[:plot_horizon, 2], label="Actual t+3")
    plt.plot(pred_raw[:plot_horizon, 2], label="Predicted t+3")
    plt.xlabel("Val sample index")
    plt.ylabel("kWh")
    plt.title(f"{house_id} | 6-step forecast | LSTM64x32 | Horizon t+3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir/f"{house_id}_6step_tplus3.png", dpi=200)
    plt.close()

    # ==========================================
    # PLOT 3: Forecast for 6 Hours Ahead (t+6)
    # Using column index 5: y_raw[:, 5]
    # ==========================================
    plt.figure(figsize=(12, 4))
    plt.plot(y_raw[:plot_horizon, 5], label="Actual t+6")
    plt.plot(pred_raw[:plot_horizon, 5], label="Predicted t+6")
    plt.xlabel("Val sample index")
    plt.ylabel("kWh")
    plt.title(f"{house_id} | 6-step forecast | LSTM64x32 | Horizon t+6")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir/f"{house_id}_6step_tplus6.png", dpi=200)
    plt.close()

    

    print(f"[{i}/{len(house_ids)}] Finished {house_id} | Val mean RMSE across horizons: {metrics['mean_rmse_across_horizons']:.6f}")

results_df = pd.DataFrame(results)
if results_df.empty:
    raise ValueError("No houses were evaluated. Check windowing or splits.")

print("\nPer-house LSTM64x32 results:")
print(results_df.head())

print("\nLSTM64x32 summary:")
print("Mean RMSE across horizons:", results_df["mean_rmse_across_horizons"].mean())
print("Median RMSE across horizons:", results_df["mean_rmse_across_horizons"].median())
print("Mean MAE across horizons:", results_df["mean_mae_across_horizons"].mean())
print("Median MAE across horizons:", results_df["mean_mae_across_horizons"].median())
print("Mean RMSE at t+1", results_df["rmse_t+1"].mean())
print("Mean RMSE at t+2", results_df["rmse_t+2"].mean())
print("Mean RMSE at t+3", results_df["rmse_t+3"].mean())
print("Mean RMSE at t+4", results_df["rmse_t+4"].mean())
print("Mean RMSE at t+5", results_df["rmse_t+5"].mean())
print("Mean RMSE at t+6", results_df["rmse_t+6"].mean())
print("Houses evaluated:", results_df["house_id"].nunique())

results_df.to_csv("lstm64x32_localised_per_house_results.csv", index=False)


#summary statistics
summary_df = pd.DataFrame([{
    "model": "lstm64x32",
    "mean_rmse_across_horizons": results_df["mean_rmse_across_horizons"].mean(),
    "median_rmse_across_horizons": results_df["mean_rmse_across_horizons"].median(),
    "mean_mae_across_horizons": results_df["mean_mae_across_horizons"].mean(),
    "median_mae_across_horizons": results_df["mean_mae_across_horizons"].median(),
    "mean_rmse_t+1": results_df["rmse_t+1"].mean(),
    "mean_rmse_t+2": results_df["rmse_t+2"].mean(),
    "mean_rmse_t+3": results_df["rmse_t+3"].mean(),
    "mean_rmse_t+4": results_df["rmse_t+4"].mean(),
    "mean_rmse_t+5": results_df["rmse_t+5"].mean(),
    "mean_rmse_t+6": results_df["rmse_t+6"].mean(),

    "mean_mae_t+1": results_df["mae_t+1"].mean(),
    "mean_mae_t+2": results_df["mae_t+2"].mean(),
    "mean_mae_t+3": results_df["mae_t+3"].mean(),
    "mean_mae_t+4": results_df["mae_t+4"].mean(),
    "mean_mae_t+5": results_df["mae_t+5"].mean(),
    "mean_mae_t+6": results_df["mae_t+6"].mean(),

    "mean_generalisation_gap": results_df["generalisation_gap"].mean(),
    "median_generalisation_gap": results_df["generalisation_gap"].median(),
    "mean_best_epoch": results_df["best_epoch"].mean(),
    "mean_epochs_run": results_df["epochs_run"].mean(),
    "n_houses": results_df["house_id"].nunique()
}])

summary_df.to_csv("lstm64x32_localised_summary.csv", index=False)
print(summary_df)

