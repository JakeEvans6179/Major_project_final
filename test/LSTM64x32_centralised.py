from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


import random
import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

'''
LSTM64x32 centralised test screening on final cohort
'''



data_path = Path("final_locked_100_normalised.parquet")

max_min_path = Path("global_weather_scaler.csv")

local_kwh_scaling = Path("local_kwh_scaler.csv")

model = load_model("LSTM64x32_global.keras")


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



df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)   #load global weather scalers and local kwh scalers

house_ids = sorted(df["LCLid"].unique())



test_data = {}
    


for i, house_id in enumerate(house_ids, start = 1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    _, _, test_df = Helper_functions.get_house_split(df, house_id, feature_cols)

    
    house_x_test, house_y_test = Helper_functions.make_xy(test_df, window_size=WINDOW_SIZE, target_col=TARGET_COL, horizon = HORIZON)
    

    if len(house_x_test) == 0:
        print(f"Skipping {house_id}: insufficient samples after windowing")
        continue

    

    #store x_val, y_val and min max kwh values in dictionary sorted by house_id

    test_data[house_id] = {
        "x_test": house_x_test,
        "y_test": house_y_test,
        "kwh_min": kwh_min,
        "kwh_max": kwh_max
    }


print("Number of houses in test_data:", len(test_data))





print("Evaluating model performance per household")

results = []

for i, house_id in enumerate(test_data.keys(), start = 1):
    print(f"Evaluatiing {i}/{len(test_data)}: {house_id}")

    pred_scaled = model.predict(test_data[house_id]["x_test"], verbose=0) #run inference 

    y_scaled = test_data[house_id]["y_test"]

    kwh_min = float(test_data[house_id]["kwh_min"])
    kwh_max = float(test_data[house_id]["kwh_max"])


    metrics, y_raw, pred_raw = Helper_functions.evaluate_predictions_multistep(
        y_scaled=y_scaled,
        pred_scaled=pred_scaled,
        min_val=kwh_min,
        max_val=kwh_max
    )


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
        "mean_mae_across_horizons": metrics["mean_mae_across_horizons"]
    })

results_df = pd.DataFrame(results)

print("Mean RMSE across horizons:", results_df["mean_rmse_across_horizons"].mean())
print("Median RMSE across horizons:", results_df["mean_rmse_across_horizons"].median())
print("Std RMSE across horizons:", results_df["mean_rmse_across_horizons"].std())
print("Mean MAE across horizons:", results_df["mean_mae_across_horizons"].mean())
print("Mean RMSE t+1:", results_df["rmse_t+1"].mean())
print("Mean RMSE t+2:", results_df["rmse_t+2"].mean())
print("Mean RMSE t+3:", results_df["rmse_t+3"].mean())
print("Mean RMSE t+4:", results_df["rmse_t+4"].mean())
print("Mean RMSE t+5:", results_df["rmse_t+5"].mean())
print("Mean RMSE t+6:", results_df["rmse_t+6"].mean())
print("Houses evaluated:", results_df["house_id"].nunique())

results_df.to_csv("centralised_LSTM64x32_per_house_test_eval.csv", index=False)



#summary statistics
summary_df = pd.DataFrame([{
    "model": "centralised_LSTM64x32",
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
}])

summary_df.to_csv("centralised_LSTM64x32_test_summary.csv", index=False)
print(summary_df)






