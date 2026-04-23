from pathlib import Path
import warnings
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import Helper_functions

"""
SARIMA 6-hour benchmark for unseen houses with limited local data,
using previously selected per-household SARIMA orders.

Setup:
- fixed 14-day window per house
- 5 days train
- 2 days val
- 7 days test

Workflow:
1. Load unseen-house limited-data parquet and scaler files
2. Load per-house selected SARIMA orders from CSV
3. For each house:
   - recover raw kWh from per-house min-max scaling
   - split into train / val / test
   - fit SARIMA once using saved best order
   - save validation predictions for ensemble alpha search
   - extend through validation observations
   - evaluate on test using rolling recursive 6-step forecasts
4. Save per-house results, summary CSV, and plots

IMPORTANT:
This assumes the saved SARIMA orders were selected on the SAME blind unseen-house setup.
Do not reuse orders from a different cohort/split unless you are intentionally doing so.
"""

plots_dir = Path("house_plots_SARIMA_unseen_6hr")
plots_dir.mkdir(exist_ok=True)

pred_dir = Path("blind_sarima_preds")
pred_dir.mkdir(exist_ok=True)

data_path = Path("unseen_2week_normalised.parquet")
max_min_path = Path("unseen_global_weather_scaler.csv")
local_kwh_scaling = Path("unseen_local_kwh_scaler.csv")

# Change this to your existing SARIMA results/orders file
orders_csv_path = Path("SARIMA_unseen_2week_per_house_results_6hr.csv")

TARGET_COL = "kwh"
HORIZON = 6


def evaluate_raw_predictions_multistep(y_true, y_pred):
    n_horizons = y_true.shape[1]
    metrics = {}

    rmse_list = []
    mae_list = []

    for h in range(n_horizons):
        y_h = y_true[:, h]
        pred_h = y_pred[:, h]

        rmse_h = np.sqrt(mean_squared_error(y_h, pred_h))
        mae_h = mean_absolute_error(y_h, pred_h)

        metrics[f"rmse_t+{h+1}"] = float(rmse_h)
        metrics[f"mae_t+{h+1}"] = float(mae_h)

        rmse_list.append(rmse_h)
        mae_list.append(mae_h)

    metrics["mean_rmse_across_horizons"] = float(np.mean(rmse_list))
    metrics["mean_mae_across_horizons"] = float(np.mean(mae_list))

    return metrics


def get_house_raw_target_splits(df, house_id, kwh_min, kwh_max):
    house_df = (
        df[df["LCLid"] == house_id]
        .copy()
        .sort_values("DateTime")
    )

    train_df = house_df[house_df["split"] == "train"].copy()
    val_df = house_df[house_df["split"] == "val"].copy()
    test_df = house_df[house_df["split"] == "test"].copy()

    train_raw = Helper_functions.unscale(train_df[TARGET_COL].to_numpy(), kwh_min, kwh_max)
    val_raw = Helper_functions.unscale(val_df[TARGET_COL].to_numpy(), kwh_min, kwh_max)
    test_raw = Helper_functions.unscale(test_df[TARGET_COL].to_numpy(), kwh_min, kwh_max)

    train_series = pd.Series(
        train_raw,
        index=pd.DatetimeIndex(train_df["DateTime"]),
        dtype=float
    ).asfreq("h")

    val_series = pd.Series(
        val_raw,
        index=pd.DatetimeIndex(val_df["DateTime"]),
        dtype=float
    ).asfreq("h")

    test_series = pd.Series(
        test_raw,
        index=pd.DatetimeIndex(test_df["DateTime"]),
        dtype=float
    ).asfreq("h")

    return train_series, val_series, test_series


def fit_sarima(train_series, order, seasonal_order):
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        fitted = model.fit(
            disp=False,
            maxiter=50,
            cov_type="none"
        )

    return fitted


def rolling_multi_step_forecast(fitted_res, eval_series, horizon=6):
    """
    Rolling recursive multi-step forecast.

    At each origin:
    - forecast next `horizon` steps
    - store true next `horizon` values
    - update the model with actual next observation only
    """
    eval_values = np.asarray(eval_series, dtype=float)

    y_true = []
    y_pred = []

    current_res = fitted_res

    n_origins = len(eval_values) - horizon + 1
    if n_origins <= 0:
        return np.empty((0, horizon)), np.empty((0, horizon))

    for i in range(n_origins):
        forecast = current_res.forecast(steps=horizon)
        forecast = np.asarray(forecast, dtype=float)
        forecast = np.clip(forecast, a_min=0, a_max=None)

        true_future = eval_values[i:i + horizon]

        y_pred.append(forecast)
        y_true.append(true_future)

        current_res = current_res.extend(np.asarray([eval_values[i]], dtype=float))

    return np.array(y_true), np.array(y_pred)


def parse_tuple_string(value):
    """
    Parse tuple-like strings such as:
    '(1, 0, 1)' or '(1, 1, 0, 24)'
    into actual Python tuples.
    """
    if isinstance(value, tuple):
        return value
    if pd.isna(value):
        raise ValueError("Order value is NaN.")
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, tuple):
        raise ValueError(f"Expected tuple-like string, got: {value}")
    return parsed


# ============================================================
# Load data
# ============================================================

df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)
)

house_ids = sorted(df["LCLid"].unique())

orders_df = pd.read_csv(orders_csv_path)
orders_df["house_id"] = orders_df["house_id"].astype(str)

required_order_cols = {"house_id", "selected_order", "selected_seasonal_order"}
missing_order_cols = required_order_cols - set(orders_df.columns)
if missing_order_cols:
    raise ValueError(
        f"Orders CSV is missing required columns: {missing_order_cols}. "
        f"Expected at least: {required_order_cols}"
    )

results = []

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    try:
        match = orders_df.loc[orders_df["house_id"] == house_id]
        if match.empty:
            print(f"Skipping {house_id}: no saved SARIMA order found.")
            continue

        saved_order = parse_tuple_string(match.iloc[0]["selected_order"])
        saved_seasonal_order = parse_tuple_string(match.iloc[0]["selected_seasonal_order"])

        kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)

        train_series, val_series, test_series = get_house_raw_target_splits(
            df, house_id, kwh_min, kwh_max
        )

        if len(train_series) == 0 or len(val_series) == 0 or len(test_series) == 0:
            print(f"Skipping {house_id}: empty split.")
            continue

        fitted_final = fit_sarima(train_series, saved_order, saved_seasonal_order)

        # Validation predictions for ensemble alpha search
        y_val_true, y_val_pred = rolling_multi_step_forecast(
            fitted_final, val_series, horizon=HORIZON
        )

        if len(y_val_true) == 0:
            print(f"Skipping {house_id}: validation series too short for 6-step forecasting.")
            continue

        val_metrics = evaluate_raw_predictions_multistep(y_val_true, y_val_pred)

        np.savez_compressed(
            pred_dir / f"{house_id}_sarima_val.npz",
            y_true_raw=y_val_true,
            pred_raw=y_val_pred,
        )

        # Extend through full validation block before evaluating on test
        fitted_after_val = fitted_final.extend(np.asarray(val_series, dtype=float))

        y_test_true, y_test_pred = rolling_multi_step_forecast(
            fitted_after_val, test_series, horizon=HORIZON
        )

        if len(y_test_true) == 0:
            print(f"Skipping {house_id}: test series too short for 6-step forecasting.")
            continue

        test_metrics = evaluate_raw_predictions_multistep(y_test_true, y_test_pred)

        np.savez_compressed(
            pred_dir / f"{house_id}_sarima_test.npz",
            y_true_raw=y_test_true,
            pred_raw=y_test_pred,
        )

        results.append({
            "house_id": house_id,
            "rmse_t+1": test_metrics["rmse_t+1"],
            "rmse_t+2": test_metrics["rmse_t+2"],
            "rmse_t+3": test_metrics["rmse_t+3"],
            "rmse_t+4": test_metrics["rmse_t+4"],
            "rmse_t+5": test_metrics["rmse_t+5"],
            "rmse_t+6": test_metrics["rmse_t+6"],
            "mae_t+1": test_metrics["mae_t+1"],
            "mae_t+2": test_metrics["mae_t+2"],
            "mae_t+3": test_metrics["mae_t+3"],
            "mae_t+4": test_metrics["mae_t+4"],
            "mae_t+5": test_metrics["mae_t+5"],
            "mae_t+6": test_metrics["mae_t+6"],
            "mean_rmse_across_horizons": test_metrics["mean_rmse_across_horizons"],
            "mean_mae_across_horizons": test_metrics["mean_mae_across_horizons"],
            "selected_order": str(saved_order),
            "selected_seasonal_order": str(saved_seasonal_order),
            "selected_val_mean_rmse": val_metrics["mean_rmse_across_horizons"],
            "selected_val_mean_mae": val_metrics["mean_mae_across_horizons"],
            "aic_final_train_fit": getattr(fitted_final, "aic", np.nan),
            "bic_final_train_fit": getattr(fitted_final, "bic", np.nan),
            "n_train_points": len(train_series),
            "n_val_points": len(val_series),
            "n_test_points": len(test_series),
        })

        plot_horizon = min(7 * 24, len(y_test_true))

        plt.figure(figsize=(12, 4))
        plt.plot(y_test_true[:plot_horizon, 0], label="Actual t+1")
        plt.plot(y_test_pred[:plot_horizon, 0], label="Predicted t+1")
        plt.xlabel("Test sample index")
        plt.ylabel("kWh")
        plt.title(f"{house_id} | SARIMA | TEST | Horizon t+1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{house_id}_tplus1.png", dpi=200)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(y_test_true[:plot_horizon, 2], label="Actual t+3")
        plt.plot(y_test_pred[:plot_horizon, 2], label="Predicted t+3")
        plt.xlabel("Test sample index")
        plt.ylabel("kWh")
        plt.title(f"{house_id} | SARIMA | TEST | Horizon t+3")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{house_id}_tplus3.png", dpi=200)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(y_test_true[:plot_horizon, 5], label="Actual t+6")
        plt.plot(y_test_pred[:plot_horizon, 5], label="Predicted t+6")
        plt.xlabel("Test sample index")
        plt.ylabel("kWh")
        plt.title(f"{house_id} | SARIMA | TEST | Horizon t+6")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{house_id}_tplus6.png", dpi=200)
        plt.close()

        print(
            f"[{i}/{len(house_ids)}] Finished {house_id} | "
            f"Using saved {saved_order} x {saved_seasonal_order} | "
            f"Test mean RMSE across horizons: {test_metrics['mean_rmse_across_horizons']:.6f}"
        )

    except Exception as e:
        print(f"Failed on {house_id}: {e}")

    finally:
        pd.DataFrame(results).to_csv(
            "SARIMA_unseen_2week_per_house_results_6hr_from_saved_orders_partial.csv",
            index=False
        )

# ============================================================
# Save outputs
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No houses were evaluated. Check saved orders CSV and splits.")

print("\nPer-house SARIMA unseen-house 6hr results:")
print(results_df.head())

print("\nSARIMA unseen-house 6hr summary:")
print("Mean RMSE across horizons:", results_df["mean_rmse_across_horizons"].mean())
print("Median RMSE across horizons:", results_df["mean_rmse_across_horizons"].median())
print("Mean MAE across horizons:", results_df["mean_mae_across_horizons"].mean())
print("Median MAE across horizons:", results_df["mean_mae_across_horizons"].median())
print("Mean RMSE at t+1:", results_df["rmse_t+1"].mean())
print("Mean RMSE at t+2:", results_df["rmse_t+2"].mean())
print("Mean RMSE at t+3:", results_df["rmse_t+3"].mean())
print("Mean RMSE at t+4:", results_df["rmse_t+4"].mean())
print("Mean RMSE at t+5:", results_df["rmse_t+5"].mean())
print("Mean RMSE at t+6:", results_df["rmse_t+6"].mean())
print("Houses evaluated:", results_df["house_id"].nunique())

results_df.to_csv("SARIMA_unseen_2week_per_house_results_6hr_from_saved_orders.csv", index=False)

summary_df = pd.DataFrame([{
    "model": "SARIMA_unseen_2week_6hr_from_saved_orders",
    "orders_source_csv": str(orders_csv_path),
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

summary_df.to_csv("SARIMA_unseen_2week_summary_6hr_from_saved_orders.csv", index=False)
print("\nFinal SARIMA unseen-house 6hr summary dataframe:")
print(summary_df)