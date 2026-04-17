from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import Helper_functions


"""
SARIMA 6-hour benchmark for household load forecasting

Workflow:
1. Load same parquet dataset and scaler files
2. For each house:
   - recover raw kWh from per-house min-max scaling
   - split into train / val / test
   - fit SARIMA candidates on train
   - select best candidate using rolling recursive 6-step validation forecasts
     on a shortened validation search window
   - refit best candidate on full train set
   - evaluate on validation using rolling recursive 6-step forecasts
3. Save per-house results, candidate search results, summary CSV, and plots
"""


plots_dir = Path("house_plots_SARIMA_6hr")
plots_dir.mkdir(exist_ok=True)

data_path = Path("../03_feature_engineering/final_locked_100_normalised.parquet")

max_min_path = Path("../03_feature_engineering/global_weather_scaler.csv")

local_kwh_scaling = Path("../03_feature_engineering/local_kwh_scaler.csv")

TARGET_COL = "kwh"
HORIZON = 6

# Use 4 weeks of validation for candidate search
VAL_SEARCH_HOURS = 24 * 28

# Candidate grid
p_values = [0, 1]
d_values = [0, 1]
q_values = [0, 1]

P_values = [0, 1]
D_value = 1
Q_values = [0, 1]
SEASONAL_PERIOD = 24


def evaluate_raw_predictions_multistep(y_true, y_pred):
    """
    Evaluate raw kWh multi-step predictions.

    y_true: shape (n_samples, horizon)
    y_pred: shape (n_samples, horizon)
    """
    n_horizons = y_true.shape[1]
    metrics = {}

    rmse_list = []
    mae_list = []

    for h in range(n_horizons):
        y_h = y_true[:, h]
        pred_h = y_pred[:, h]

        rmse_h = np.sqrt(mean_squared_error(y_h, pred_h))
        mae_h = mean_absolute_error(y_h, pred_h)

        metrics[f"rmse_t+{h+1}"] = rmse_h
        metrics[f"mae_t+{h+1}"] = mae_h

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
    #test_df = house_df[house_df["split"] == "test"].copy()

    train_raw = Helper_functions.unscale(train_df[TARGET_COL].to_numpy(), kwh_min, kwh_max)
    val_raw = Helper_functions.unscale(val_df[TARGET_COL].to_numpy(), kwh_min, kwh_max)
    #test_raw = Helper_functions.unscale(test_df[TARGET_COL].to_numpy(), kwh_min, kwh_max)

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

    #test_series = pd.Series(
        #test_raw,
        #index=pd.DatetimeIndex(test_df["DateTime"]),
        #dtype=float
    #).asfreq("h")

    return train_series, val_series


def build_candidate_grid():
    candidates = []
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for Q in Q_values:
                        order = (p, d, q)
                        seasonal_order = (P, D_value, Q, SEASONAL_PERIOD)
                        candidates.append((order, seasonal_order))
    return candidates


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
    - store the true next `horizon` values
    - update the model with the actual next observation only
    - continue

    Returns:
        y_true: (n_samples, horizon)
        y_pred: (n_samples, horizon)
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

        # Physical constraint: demand cannot be negative
        forecast = np.clip(forecast, a_min=0, a_max=None)

        true_future = eval_values[i:i + horizon]

        y_pred.append(forecast)
        y_true.append(true_future)

        # Update with actual next single observation only
        current_res = current_res.extend(np.asarray([eval_values[i]], dtype=float))

    return np.array(y_true), np.array(y_pred)


def search_best_sarima(train_series, val_series_search, house_id=None):
    candidates = build_candidate_grid()

    search_rows = []
    best = None

    for idx, (order, seasonal_order) in enumerate(candidates, start=1):
        try:
            fitted = fit_sarima(train_series, order, seasonal_order)
            y_val_true, y_val_pred = rolling_multi_step_forecast(
                fitted, val_series_search, horizon=HORIZON
            )

            if len(y_val_true) == 0:
                raise ValueError("Validation series too short for 6-step forecasting.")

            val_metrics = evaluate_raw_predictions_multistep(y_val_true, y_val_pred)

            row = {
                "house_id": house_id,
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "val_rmse_t+1": val_metrics["rmse_t+1"],
                "val_rmse_t+2": val_metrics["rmse_t+2"],
                "val_rmse_t+3": val_metrics["rmse_t+3"],
                "val_rmse_t+4": val_metrics["rmse_t+4"],
                "val_rmse_t+5": val_metrics["rmse_t+5"],
                "val_rmse_t+6": val_metrics["rmse_t+6"],

                "val_mae_t+1": val_metrics["mae_t+1"],
                "val_mae_t+2": val_metrics["mae_t+2"],
                "val_mae_t+3": val_metrics["mae_t+3"],
                "val_mae_t+4": val_metrics["mae_t+4"],
                "val_mae_t+5": val_metrics["mae_t+5"],
                "val_mae_t+6": val_metrics["mae_t+6"],

                "val_mean_rmse_across_horizons": val_metrics["mean_rmse_across_horizons"],
                "val_mean_mae_across_horizons": val_metrics["mean_mae_across_horizons"],
                "aic_train_fit": getattr(fitted, "aic", np.nan),
                "bic_train_fit": getattr(fitted, "bic", np.nan),
                "search_rank_index": idx,
                "status": "ok",
                "error_message": "",
            }
            search_rows.append(row)

            if best is None or val_metrics["mean_rmse_across_horizons"] < best["val_mean_rmse_across_horizons"]:
                best = {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "val_rmse_t+1": val_metrics["rmse_t+1"],
                    "val_rmse_t+2": val_metrics["rmse_t+2"],
                    "val_rmse_t+3": val_metrics["rmse_t+3"],
                    "val_rmse_t+4": val_metrics["rmse_t+4"],
                    "val_rmse_t+5": val_metrics["rmse_t+5"],
                    "val_rmse_t+6": val_metrics["rmse_t+6"],
                    "val_mae_t+1": val_metrics["mae_t+1"],
                    "val_mae_t+2": val_metrics["mae_t+2"],
                    "val_mae_t+3": val_metrics["mae_t+3"],
                    "val_mae_t+4": val_metrics["mae_t+4"],
                    "val_mae_t+5": val_metrics["mae_t+5"],
                    "val_mae_t+6": val_metrics["mae_t+6"],
                    "val_mean_rmse_across_horizons": val_metrics["mean_rmse_across_horizons"],
                    "val_mean_mae_across_horizons": val_metrics["mean_mae_across_horizons"],
                    "aic_train_fit": getattr(fitted, "aic", np.nan),
                    "bic_train_fit": getattr(fitted, "bic", np.nan),
                }

        except Exception as e:
            print(
                f"Candidate failed for house {house_id} | "
                f"order={order} seasonal={seasonal_order} | {repr(e)}"
            )

            search_rows.append({
                "house_id": house_id,
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "val_rmse_t+1": np.nan,
                "val_rmse_t+2": np.nan,
                "val_rmse_t+3": np.nan,
                "val_rmse_t+4": np.nan,
                "val_rmse_t+5": np.nan,
                "val_rmse_t+6": np.nan,
                "val_mae_t+1": np.nan,
                "val_mae_t+2": np.nan,
                "val_mae_t+3": np.nan,
                "val_mae_t+4": np.nan,
                "val_mae_t+5": np.nan,
                "val_mae_t+6": np.nan,
                "val_mean_rmse_across_horizons": np.nan,
                "val_mean_mae_across_horizons": np.nan,
                "aic_train_fit": np.nan,
                "bic_train_fit": np.nan,
                "search_rank_index": idx,
                "status": "failed",
                "error_message": str(e),
            })

    search_df = pd.DataFrame(search_rows)
    return best, search_df


# ============================================================
# Main benchmark loop
# ============================================================

df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)
)

# house_ids = sorted(df["LCLid"].unique())[:5]
house_ids = sorted(df["LCLid"].unique())

results = []
all_search_dfs = []

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    try:
        kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)

        train_series, val_series = get_house_raw_target_splits(
            df, house_id, kwh_min, kwh_max
        )

        val_series_search = val_series.iloc[:VAL_SEARCH_HOURS] if len(val_series) > VAL_SEARCH_HOURS else val_series

        if len(train_series) == 0 or len(val_series_search) == 0:
            print(f"Skipping {house_id}: empty split.")

        else:
            best_cfg, search_df = search_best_sarima(
                train_series, val_series_search, house_id=house_id
            )
            all_search_dfs.append(search_df)

            if best_cfg is None:
                print(f"Skipping {house_id}: all SARIMA candidates failed.")

            else:
                best_order = best_cfg["order"]
                best_seasonal_order = best_cfg["seasonal_order"]

                #fit on training data 
                fitted_final = fit_sarima(train_series, best_order, best_seasonal_order)

                #evaluate on validation data 
                y_val_true, y_val_pred = rolling_multi_step_forecast(
                    fitted_final, val_series, horizon=HORIZON
                )

                if len(y_val_true) == 0:
                    print(f"Skipping {house_id}: validation series too short for 6-step forecasting.")
                    continue

                val_metrics = evaluate_raw_predictions_multistep(y_val_true, y_val_pred)

                n_candidates_total = len(search_df)
                n_candidates_success = int((search_df["status"] == "ok").sum())
                n_candidates_failed = int((search_df["status"] == "failed").sum())

                results.append({
                    "house_id": house_id,
                    "rmse_t+1": val_metrics["rmse_t+1"],
                    "rmse_t+2": val_metrics["rmse_t+2"],
                    "rmse_t+3": val_metrics["rmse_t+3"],
                    "rmse_t+4": val_metrics["rmse_t+4"],
                    "rmse_t+5": val_metrics["rmse_t+5"],
                    "rmse_t+6": val_metrics["rmse_t+6"],
                    "mae_t+1": val_metrics["mae_t+1"],
                    "mae_t+2": val_metrics["mae_t+2"],
                    "mae_t+3": val_metrics["mae_t+3"],
                    "mae_t+4": val_metrics["mae_t+4"],
                    "mae_t+5": val_metrics["mae_t+5"],
                    "mae_t+6": val_metrics["mae_t+6"],
                    "mean_rmse_across_horizons": val_metrics["mean_rmse_across_horizons"],
                    "mean_mae_across_horizons": val_metrics["mean_mae_across_horizons"],
                    "selected_order": str(best_order),
                    "selected_seasonal_order": str(best_seasonal_order),
                    "val_mean_rmse_of_selected": best_cfg["val_mean_rmse_across_horizons"],
                    "val_mean_mae_of_selected": best_cfg["val_mean_mae_across_horizons"],
                    "aic_train_fit_of_selected": best_cfg["aic_train_fit"],
                    "bic_train_fit_of_selected": best_cfg["bic_train_fit"],
                    "aic_final_train_refit": getattr(fitted_final, "aic", np.nan),
                    "bic_final_train_refit": getattr(fitted_final, "bic", np.nan),
                    "n_train_points": len(train_series),
                    "n_val_points": len(val_series),
                    "n_val_search_points": len(val_series_search),
                    "n_candidates_total": n_candidates_total,
                    "n_candidates_success": n_candidates_success,
                    "n_candidates_failed": n_candidates_failed,
                })

                plot_horizon = min(14 * 24, len(y_val_true))

                # t+1
                plt.figure(figsize=(12, 4))
                plt.plot(y_val_true[:plot_horizon, 0], label="Actual t+1")
                plt.plot(y_val_pred[:plot_horizon, 0], label="Predicted t+1")
                plt.xlabel("Val sample index")
                plt.ylabel("kWh")
                plt.title(f"{house_id} | SARIMA | Horizon t+1")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"{house_id}_tplus1.png", dpi=200)
                plt.close()

                # t+3
                plt.figure(figsize=(12, 4))
                plt.plot(y_val_true[:plot_horizon, 2], label="Actual t+3")
                plt.plot(y_val_pred[:plot_horizon, 2], label="Predicted t+3")
                plt.xlabel("Val sample index")
                plt.ylabel("kWh")
                plt.title(f"{house_id} | SARIMA | Horizon t+3")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"{house_id}_tplus3.png", dpi=200)
                plt.close()

                # t+6
                plt.figure(figsize=(12, 4))
                plt.plot(y_val_true[:plot_horizon, 5], label="Actual t+6")
                plt.plot(y_val_pred[:plot_horizon, 5], label="Predicted t+6")
                plt.xlabel("Val sample index")
                plt.ylabel("kWh")
                plt.title(f"{house_id} | SARIMA | Horizon t+6")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"{house_id}_tplus6.png", dpi=200)
                plt.close()

                print(
                    f"[{i}/{len(house_ids)}] Finished {house_id} | "
                    f"Selected {best_order} x {best_seasonal_order} | "
                    f"Mean RMSE across horizons: {val_metrics['mean_rmse_across_horizons']:.6f}"
                )

    except Exception as e:
        print(f"Failed on {house_id}: {e}")

    finally:
        pd.DataFrame(results).to_csv("SARIMA_per_house_results_6hr_partial.csv", index=False)

        if all_search_dfs:
            pd.concat(all_search_dfs, ignore_index=True).to_csv(
                "SARIMA_candidate_search_results_6hr_partial.csv",
                index=False
            )

# ============================================================
# Save outputs
# ============================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError("No houses were evaluated. Check SARIMA fit/search settings.")

search_results_df = pd.concat(all_search_dfs, ignore_index=True)

print("\nPer-house SARIMA 6hr results:")
print(results_df.head())

print("\nSARIMA 6hr summary:")
print("Mean RMSE across horizons:", results_df["mean_rmse_across_horizons"].mean())
print("Mean MAE across horizons:", results_df["mean_mae_across_horizons"].mean())
print("Mean RMSE at t+1:", results_df["rmse_t+1"].mean())
print("Mean RMSE at t+2:", results_df["rmse_t+2"].mean())
print("Mean RMSE at t+3:", results_df["rmse_t+3"].mean())
print("Mean RMSE at t+4:", results_df["rmse_t+4"].mean())
print("Mean RMSE at t+5:", results_df["rmse_t+5"].mean())
print("Mean RMSE at t+6:", results_df["rmse_t+6"].mean())
print("Houses evaluated:", results_df["house_id"].nunique())

results_df.to_csv("SARIMA_per_house_results_6hr.csv", index=False)
search_results_df.to_csv("SARIMA_candidate_search_results_6hr.csv", index=False)

summary_df = pd.DataFrame([{
    "model": "SARIMA_6hr",
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

summary_df.to_csv("SARIMA_summary_6hr.csv", index=False)
print("\nFinal SARIMA 6hr summary dataframe:")
print(summary_df)