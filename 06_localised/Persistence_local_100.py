from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import Helper_functions

"""
6-hour persistence baseline -validation screening

Definition:
For each sample, predict the next 6 hours as equal to the most recent
observed kWh value at the end of the 24-hour input window.

This is the direct multi-step persistence baseline aligned with the
6-step neural models.
"""



data_path = Path("../03_feature_engineering/final_locked_100_normalised.parquet")

max_min_path = Path("../03_feature_engineering/global_weather_scaler.csv")

local_kwh_scaling = Path("../03_feature_engineering/local_kwh_scaler.csv")

feature_cols = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
]

WINDOW_SIZE = 24
HORIZON = 6
TARGET_COL = "kwh"

df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)
)

# house_ids = sorted(df["LCLid"].unique())[:5]
house_ids = sorted(df["LCLid"].unique())

results = []

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    try:
        kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
        _, val_df, _ = Helper_functions.get_house_split(df, house_id, feature_cols)

        val_kwh = val_df[TARGET_COL].to_numpy(dtype=np.float32)

        

        # Need enough points for a 24-hour input window plus 6 target hours
        n_samples = len(val_kwh) - WINDOW_SIZE - HORIZON + 1
        if n_samples <= 0:
            print(f"Skipping {house_id}: insufficient validation samples")
            continue

        y_val = []
        pred_scaled = []

        for s in range(n_samples):
            # Last observed point in the input window
            last_val = val_kwh[s + WINDOW_SIZE - 1]

            # True next 6 hours
            future_vals = val_kwh[s + WINDOW_SIZE : s + WINDOW_SIZE + HORIZON]

            # Persistence prediction: repeat last observed value
            pred_vals = np.array([last_val, last_val, last_val, last_val, last_val, last_val], dtype=np.float32)

            y_val.append(future_vals)
            pred_scaled.append(pred_vals)

        y_val = np.array(y_val, dtype=np.float32)
        pred_scaled = np.array(pred_scaled, dtype=np.float32)

        metrics, y_raw, pred_raw = Helper_functions.evaluate_predictions_multistep(
            y_scaled=y_val,
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
            "mean_mae_across_horizons": metrics["mean_mae_across_horizons"],
            "n_val_samples": len(y_val),
        })

        

        print(
            f"[{i}/{len(house_ids)}] Finished {house_id} | "
            f"Mean RMSE across horizons: {metrics['mean_rmse_across_horizons']:.6f}"
        )

    except Exception as e:
        print(f"Failed on {house_id}: {e}")


results_df = pd.DataFrame(results)
if results_df.empty:
    raise ValueError("No houses were evaluated.")

print("\nPer-house persistence 6hr results:")
print(results_df.head())

print("\nPersistence 6hr summary:")
print("Mean RMSE across horizons:", results_df["mean_rmse_across_horizons"].mean())
print("Mean MAE across horizons:", results_df["mean_mae_across_horizons"].mean())
print("Mean RMSE at t+1:", results_df["rmse_t+1"].mean())
print("Mean RMSE at t+2:", results_df["rmse_t+2"].mean())
print("Mean RMSE at t+3:", results_df["rmse_t+3"].mean())
print("Mean RMSE at t+4:", results_df["rmse_t+4"].mean())
print("Mean RMSE at t+5:", results_df["rmse_t+5"].mean())
print("Mean RMSE at t+6:", results_df["rmse_t+6"].mean())
print("Houses evaluated:", results_df["house_id"].nunique())

results_df.to_csv("persistence_per_house_results.csv", index=False)

summary_df = pd.DataFrame([{
    "model": "persistence_6hr",
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
    "n_houses": results_df["house_id"].nunique()
}])

summary_df.to_csv("persistence_summary.csv", index=False)
print(summary_df)