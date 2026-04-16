from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import random
import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

'''
LSTM64x32 validation metrics

Load global model, calculate original metrics (RMSE, MAE on validation set)

Carry out fine tuning, calculate new metrics (RMSE, MAE on validation set)

Compare to see if there is any improvement
'''

data_path = Path("../../03_feature_engineering/final_locked_100_normalised.parquet")

max_min_path = Path("../../03_feature_engineering/global_weather_scaler.csv")

local_kwh_scaling = Path("../../03_feature_engineering/local_kwh_scaler.csv")

global_model_path = Path("models/LSTM64x32_global.keras" )  #load the global model


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
def compile_for_finetuning(model):
    

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
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
        batch_size=256,
        verbose=1,
        callbacks=[es]
    )

    return model, history




df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)   #load global weather scalers and local kwh scalers
house_ids = sorted(df["LCLid"].unique())


results = []
    
#print(x_val)

#convert training validation and test data into suitable format for lstm then concatenate for all households

#print(df)

for i, house_id in enumerate(house_ids, start = 1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(69)
    np.random.seed(69)
    random.seed(69)

    starting_model = load_model(global_model_path)   #copy model so starting weights are not changed for next house



    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    train_df, val_df, test_df = Helper_functions.get_house_split(df, house_id, feature_cols)

    house_x_train, house_y_train = Helper_functions.make_xy(train_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon = HORIZON)
    house_x_val, house_y_val = Helper_functions.make_xy(val_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon = HORIZON)
    
    if len(house_x_train) == 0 or len(house_x_val) == 0:
        print(f"Skipping {house_id}: insufficient samples after windowing")
        continue


    

    #run inference on global model
    pred_scaled_centralised = starting_model.predict(house_x_val, verbose=0) #run inference

    #find metrics on validation set (centralised)
    centralised_metrics, _, _ = Helper_functions.evaluate_predictions_multistep(
        y_scaled=house_y_val,
        pred_scaled=pred_scaled_centralised,
        min_val=kwh_min,
        max_val=kwh_max
    )



    
    #train model
    fine_tuned_model, history = train_model(house_x_train, house_y_train, house_x_val, house_y_val, starting_model)

    #find validation metrics

    pred_scaled_fine_tuned = fine_tuned_model.predict(house_x_val, verbose = 0)

    fine_tuned_metrics, _, _ = Helper_functions.evaluate_predictions_multistep(
        y_scaled = house_y_val,
        pred_scaled = pred_scaled_fine_tuned,
        min_val = kwh_min,
        max_val = kwh_max
    )

    delta_rmse = centralised_metrics['mean_rmse_across_horizons'] - fine_tuned_metrics["mean_rmse_across_horizons"]
    delta_mae = centralised_metrics["mean_mae_across_horizons"] - fine_tuned_metrics["mean_mae_across_horizons"]


    results.append({
        "house_id": house_id,
        "centralised_mean_rmse": centralised_metrics["mean_rmse_across_horizons"],
        "centralised_mean_mae": centralised_metrics["mean_mae_across_horizons"],
        "fine_tuned_mean_rmse": fine_tuned_metrics["mean_rmse_across_horizons"],
        "fine_tuned_mean_mae": fine_tuned_metrics["mean_mae_across_horizons"],
        "delta_rmse": delta_rmse,
        "delta_mae": delta_mae,
        "epochs_run": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1)
    })

results_df = pd.DataFrame(results)

results_df.to_csv("fine_tuned_LSTM64x32_per_house_eval.csv", index=False)

print("Mean RMSE across horizons centralised:", results_df["centralised_mean_rmse"].mean())
print("Median RMSE across horizons centralised:", results_df["centralised_mean_rmse"].median())

print("Mean MAE across horizons centralised:", results_df["centralised_mean_mae"].mean())
print("Median MAE across horizons centralised:", results_df["centralised_mean_mae"].median())

print("Mean RMSE across horizons fine tuned:", results_df["fine_tuned_mean_rmse"].mean())
print("Median RMSE across horizons fine tuned:", results_df["fine_tuned_mean_rmse"].median())

print("Mean MAE across horizons fine tuned:", results_df["fine_tuned_mean_mae"].mean())
print("Median MAE across horizons fine tuned:", results_df["fine_tuned_mean_mae"].median())


print("Mean delta RMSE across horizons:", results_df["delta_rmse"].mean())
print("Median delta RMSE across horizons:", results_df["delta_rmse"].median())

print("Mean delta MAE across horizons:", results_df["delta_mae"].mean())
print("Median delta MAE across horizons:", results_df["delta_mae"].median())

print("Houses improved in RMSE:", (results_df["delta_rmse"] > 0).sum())
print("Houses worsened in RMSE:", (results_df["delta_rmse"] < 0).sum())

print("Houses improved in MAE:", (results_df["delta_mae"] > 0).sum())
print("Houses worsened in MAE:", (results_df["delta_mae"] < 0).sum())



summary_df = pd.DataFrame([{
    "model": "fine_tuned_LSTM64x32",
    "Mean RMSE across horizons centralised": results_df["centralised_mean_rmse"].mean(),
    "Median RMSE across horizons centralised": results_df["centralised_mean_rmse"].median(),
    "Mean MAE across horizons centralised": results_df["centralised_mean_mae"].mean(),
    "Median MAE across horizons centralised": results_df["centralised_mean_mae"].median(),
    "Mean RMSE across horizons fine tuned": results_df["fine_tuned_mean_rmse"].mean(),
    "Median RMSE across horizons fine tuned": results_df["fine_tuned_mean_rmse"].median(),
    "Mean MAE across horizons fine tuned": results_df["fine_tuned_mean_mae"].mean(),
    "Median MAE across horizons fine tuned": results_df["fine_tuned_mean_mae"].median(),

    "Mean delta RMSE across horizons": results_df["delta_rmse"].mean(),
    "Median delta RMSE across horizons": results_df["delta_rmse"].median(),

    "Mean delta MAE across horizons": results_df["delta_mae"].mean(),
    "Median delta MAE across horizons": results_df["delta_mae"].median(),

    "Mean epochs run": results_df["epochs_run"].mean(),
    "Mean best epoch": results_df["best_epoch"].mean()




    
}])

summary_df.to_csv("fine_tuned_LSTM64x32_summary.csv", index=False)
print(summary_df)