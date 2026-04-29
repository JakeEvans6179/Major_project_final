from pathlib import Path
import numpy as np
import pandas as pd

"""
Create a compact summary-only CSV 
Used for fine tuning per house results file

Outputs:
- raw pre/post mean RMSE
- raw pre/post mean MAE
- raw delta RMSE / MAE
- scaled pre/post mean RMSE
- scaled delta mean RMSE

Scaled RMSE is computed as:
    scaled_rmse = raw_rmse / (kwh_max - kwh_min)
"""

# ============================================================
# CONFIG
# ============================================================

RESULTS_CSV = Path("fine_tuned_CNN_LSTM_per_house.csv")
SCALER_CSV = Path("data_files/local_kwh_scaler.csv")
OUTPUT_SUMMARY_CSV = Path("fine_tuned_CNN_LSTM_summary_compact.csv")

# Your actual columns
PRE_RMSE_COL = "federated_mean_rmse"
PRE_MAE_COL = "federated_mean_mae"
POST_RMSE_COL = "fine_tuned_mean_rmse"
POST_MAE_COL = "fine_tuned_mean_mae"
DELTA_RMSE_COL = "delta_rmse"
DELTA_MAE_COL = "delta_mae"

# ============================================================
# LOAD
# ============================================================

results_df = pd.read_csv(RESULTS_CSV)
scalers_df = pd.read_csv(SCALER_CSV)

results_df["house_id"] = results_df["house_id"].astype(str)
scalers_df["house_id"] = scalers_df["house_id"].astype(str)

required_cols = {
    "house_id",
    PRE_RMSE_COL, PRE_MAE_COL,
    POST_RMSE_COL, POST_MAE_COL,
    DELTA_RMSE_COL, DELTA_MAE_COL,
}
missing = required_cols - set(results_df.columns)
if missing:
    raise ValueError(f"Results CSV missing required columns: {missing}")

required_scaler_cols = {"house_id", "kwh_min", "kwh_max"}
missing_scalers = required_scaler_cols - set(scalers_df.columns)
if missing_scalers:
    raise ValueError(f"Scaler CSV missing required columns: {missing_scalers}")

# ============================================================
# MERGE SCALERS
# ============================================================

df = results_df.merge(
    scalers_df[["house_id", "kwh_min", "kwh_max"]],
    on="house_id",
    how="left"
)

if df[["kwh_min", "kwh_max"]].isna().any().any():
    missing_ids = df.loc[df["kwh_min"].isna() | df["kwh_max"].isna(), "house_id"].tolist()
    raise ValueError(f"Missing scaler rows for houses: {missing_ids}")

df["kwh_range"] = df["kwh_max"] - df["kwh_min"]

if (df["kwh_range"] <= 0).any():
    bad_ids = df.loc[df["kwh_range"] <= 0, "house_id"].tolist()
    raise ValueError(f"Non-positive kwh range for houses: {bad_ids}")

# ============================================================
# COMPUTE SCALED RMSE
# ============================================================

df["centralised_scaled_mean_rmse"] = df[PRE_RMSE_COL] / df["kwh_range"]
df["fine_tuned_scaled_mean_rmse"] = df[POST_RMSE_COL] / df["kwh_range"]
df["delta_scaled_mean_rmse"] = (
    df["centralised_scaled_mean_rmse"] - df["fine_tuned_scaled_mean_rmse"]
)

# ============================================================
# BUILD SUMMARY
# ============================================================

summary_df = pd.DataFrame([{
    "model": RESULTS_CSV.stem,

    "mean_centralised_rmse": df[PRE_RMSE_COL].mean(),
    "median_centralised_rmse": df[PRE_RMSE_COL].median(),
    "mean_centralised_mae": df[PRE_MAE_COL].mean(),
    "median_centralised_mae": df[PRE_MAE_COL].median(),

    "mean_fine_tuned_rmse": df[POST_RMSE_COL].mean(),
    "median_fine_tuned_rmse": df[POST_RMSE_COL].median(),
    "mean_fine_tuned_mae": df[POST_MAE_COL].mean(),
    "median_fine_tuned_mae": df[POST_MAE_COL].median(),

    "mean_delta_rmse": df[DELTA_RMSE_COL].mean(),
    "median_delta_rmse": df[DELTA_RMSE_COL].median(),
    "mean_delta_mae": df[DELTA_MAE_COL].mean(),
    "median_delta_mae": df[DELTA_MAE_COL].median(),

    "mean_centralised_scaled_rmse": df["centralised_scaled_mean_rmse"].mean(),
    "median_centralised_scaled_rmse": df["centralised_scaled_mean_rmse"].median(),

    "mean_fine_tuned_scaled_rmse": df["fine_tuned_scaled_mean_rmse"].mean(),
    "median_fine_tuned_scaled_rmse": df["fine_tuned_scaled_mean_rmse"].median(),

    "mean_delta_scaled_rmse": df["delta_scaled_mean_rmse"].mean(),
    "median_delta_scaled_rmse": df["delta_scaled_mean_rmse"].median(),

    "mean_epochs_run": df["epochs_run"].mean() if "epochs_run" in df.columns else np.nan,
    "median_best_epoch": df["best_epoch"].median() if "best_epoch" in df.columns else np.nan,

    "n_houses": int(df["house_id"].nunique()),
}])

summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

print(summary_df)
print(f"\nSaved summary to: {OUTPUT_SUMMARY_CSV}")