from pathlib import Path
import numpy as np
import pandas as pd

import Helper_functions

"""
Create a summary-only CSV with:
- mean / median RMSE
- mean / median MAE
- mean / median scaled RMSE (per-house min-max range)
- mean / median NRMSE_std
- mean / median NRMSE_mean

Definitions:
- scaled RMSE   = RMSE / (kwh_max - kwh_min)
- NRMSE_std     = RMSE / std(raw evaluation series)
- NRMSE_mean    = RMSE / mean(raw evaluation series)

Used for centralised screening and localised per house files only
"""

# ============================================================
# CONFIG
# ============================================================

RESULTS_CSV = Path("cnn_lstm_localised_per_house_results.csv")
OUTPUT_SUMMARY_CSV = Path("localised_CNN_LSTM_summary_with_nrmse.csv")

DATA_PATH = Path("data_files/final_locked_100_normalised.parquet")
MAX_MIN_PATH = Path("data_files/global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("data_files/local_kwh_scaler.csv")

FEATURE_COLS = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
]

# Which split the original CSV was evaluated on
EVAL_SPLIT = "val"

# Main metric columns in the results CSV
RMSE_COL = "mean_rmse_across_horizons"
MAE_COL = "mean_mae_across_horizons"

# For fine-tuning CSV instead, use for example:
# RMSE_COL = "fine_tuned_mean_rmse"
# MAE_COL = "fine_tuned_mean_mae"

# ============================================================
# LOAD
# ============================================================

results_df = pd.read_csv(RESULTS_CSV)
scalers_df = pd.read_csv(LOCAL_KWH_SCALING)

required_cols = {"house_id", RMSE_COL, MAE_COL}
missing = required_cols - set(results_df.columns)
if missing:
    raise ValueError(f"Results CSV missing required columns: {missing}")

required_scaler_cols = {"house_id", "kwh_min", "kwh_max"}
missing_scalers = required_scaler_cols - set(scalers_df.columns)
if missing_scalers:
    raise ValueError(f"Scaler CSV missing required columns: {missing_scalers}")

results_df["house_id"] = results_df["house_id"].astype(str)
scalers_df["house_id"] = scalers_df["house_id"].astype(str)

df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING)
)

# ============================================================
# BUILD PER-HOUSE NORMALISED VALUES
# ============================================================

scaled_rmse_values = []
nrmse_std_values = []
nrmse_mean_values = []

for _, row in results_df.iterrows():
    house_id = str(row["house_id"])
    mean_rmse = float(row[RMSE_COL])

    scaler_row = scalers_df.loc[scalers_df["house_id"] == house_id]
    if scaler_row.empty:
        print(f"Skipping {house_id}: no scaler row found.")
        continue

    kwh_min = float(scaler_row["kwh_min"].iloc[0])
    kwh_max = float(scaler_row["kwh_max"].iloc[0])
    kwh_range = kwh_max - kwh_min

    if kwh_range <= 0:
        print(f"Skipping {house_id}: non-positive kwh range.")
        continue

    train_df, val_df, test_df = Helper_functions.get_house_split(df, house_id, FEATURE_COLS)

    if EVAL_SPLIT == "train":
        eval_df = train_df
    elif EVAL_SPLIT == "val":
        eval_df = val_df
    elif EVAL_SPLIT == "test":
        eval_df = test_df
    else:
        raise ValueError("EVAL_SPLIT must be one of: train, val, test")

    if eval_df.empty:
        print(f"Skipping {house_id}: empty {EVAL_SPLIT} split.")
        continue

    # raw unique evaluation series
    eval_raw = Helper_functions.unscale(
        eval_df["kwh"].to_numpy(),
        kwh_min,
        kwh_max
    )

    target_mean_kwh = float(np.mean(eval_raw))
    target_std_kwh = float(np.std(eval_raw, ddof=0))

    scaled_rmse = mean_rmse / kwh_range
    nrmse_std = mean_rmse / target_std_kwh if target_std_kwh > 0 else np.nan
    nrmse_mean = mean_rmse / target_mean_kwh if target_mean_kwh > 0 else np.nan

    scaled_rmse_values.append(scaled_rmse)
    nrmse_std_values.append(nrmse_std)
    nrmse_mean_values.append(nrmse_mean)

if len(scaled_rmse_values) == 0:
    raise ValueError("No valid normalized values were computed.")

# ============================================================
# BUILD SUMMARY
# ============================================================

summary_df = pd.DataFrame([{
    "model": RESULTS_CSV.stem,
    "mean_rmse_across_horizons": results_df[RMSE_COL].mean(),
    "median_rmse_across_horizons": results_df[RMSE_COL].median(),
    "mean_mae_across_horizons": results_df[MAE_COL].mean(),
    "median_mae_across_horizons": results_df[MAE_COL].median(),

    "mean_scaled_rmse": float(np.nanmean(scaled_rmse_values)),
    "median_scaled_rmse": float(np.nanmedian(scaled_rmse_values)),

    "mean_nrmse_std": float(np.nanmean(nrmse_std_values)),
    "median_nrmse_std": float(np.nanmedian(nrmse_std_values)),

    "mean_nrmse_mean": float(np.nanmean(nrmse_mean_values)),
    "median_nrmse_mean": float(np.nanmedian(nrmse_mean_values)),

    "n_houses": int(results_df["house_id"].nunique()),
    "eval_split": EVAL_SPLIT,
}])

summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

print(summary_df)
print(f"\nSaved summary to: {OUTPUT_SUMMARY_CSV}")