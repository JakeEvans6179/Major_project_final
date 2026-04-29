from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""
Weighted-average ensemble for blind unseen-house test.

Ensemble:
    y_ens = alpha * y_sarima + (1 - alpha) * y_flft

Search alpha on validation.
Evaluate best alpha on test.
"""

SARIMA_DIR = Path("blind_sarima_preds")
FLFT_DIR = Path("blind_flft_preds")

OUT_ALPHA_SEARCH = Path("blind_ensemble_alpha_search.csv")
OUT_TEST_PER_HOUSE = Path("blind_ensemble_test_per_house.csv")
OUT_TEST_SUMMARY = Path("blind_ensemble_test_summary.csv")

ALPHAS = np.arange(0.0, 1.0001, 0.05)
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


def load_common_house_ids():
    sarima_val = {
        p.name.replace("_sarima_val.npz", ""): p
        for p in SARIMA_DIR.glob("*_sarima_val.npz")
    }
    flft_val = {
        p.name.replace("_flft_val.npz", ""): p
        for p in FLFT_DIR.glob("*_flft_val.npz")
    }

    common = sorted(set(sarima_val.keys()) & set(flft_val.keys()))
    if not common:
        raise ValueError("No common households found between SARIMA and FLFT validation predictions.")

    return common


def load_pair(house_id, split):
    sarima_path = SARIMA_DIR / f"{house_id}_sarima_{split}.npz"
    flft_path = FLFT_DIR / f"{house_id}_flft_{split}.npz"

    if not sarima_path.exists():
        raise FileNotFoundError(f"Missing SARIMA file: {sarima_path}")
    if not flft_path.exists():
        raise FileNotFoundError(f"Missing FLFT file: {flft_path}")

    sarima_data = np.load(sarima_path)
    flft_data = np.load(flft_path)

    y_true_sarima = sarima_data["y_true_raw"]
    y_pred_sarima = sarima_data["pred_raw"]

    y_true_flft = flft_data["y_true_raw"]
    y_pred_flft = flft_data["pred_raw"]

    if y_true_sarima.shape != y_true_flft.shape:
        raise ValueError(f"{house_id} {split}: y_true shapes differ.")
    if y_pred_sarima.shape != y_pred_flft.shape:
        raise ValueError(f"{house_id} {split}: pred shapes differ.")
    if not np.allclose(y_true_sarima, y_true_flft):
        raise ValueError(f"{house_id} {split}: y_true arrays do not match.")

    return y_true_sarima, y_pred_sarima, y_pred_flft


# ============================================================
# Step 1: Search alpha on validation
# ============================================================

house_ids = load_common_house_ids()
alpha_rows = []

for alpha in ALPHAS:
    val_metrics_all = []

    for house_id in house_ids:
        y_true, y_pred_sarima, y_pred_flft = load_pair(house_id, "val")
        y_pred_ens = alpha * y_pred_sarima + (1.0 - alpha) * y_pred_flft

        metrics = evaluate_raw_predictions_multistep(y_true, y_pred_ens)
        val_metrics_all.append(metrics)

    val_df = pd.DataFrame(val_metrics_all)

    alpha_rows.append({
        "alpha_sarima": float(alpha),
        "alpha_flft": float(1.0 - alpha),
        "mean_rmse_across_horizons": float(val_df["mean_rmse_across_horizons"].mean()),
        "median_rmse_across_horizons": float(val_df["mean_rmse_across_horizons"].median()),
        "mean_mae_across_horizons": float(val_df["mean_mae_across_horizons"].mean()),
        "median_mae_across_horizons": float(val_df["mean_mae_across_horizons"].median()),
        "n_houses": len(house_ids),
    })

alpha_search_df = pd.DataFrame(alpha_rows).sort_values(
    "mean_rmse_across_horizons"
).reset_index(drop=True)

alpha_search_df.to_csv(OUT_ALPHA_SEARCH, index=False)

best_alpha = float(alpha_search_df.iloc[0]["alpha_sarima"])
print("Best validation alpha:")
print(alpha_search_df.iloc[0])

# ============================================================
# Step 2: Evaluate best alpha on test
# ============================================================

test_rows = []

for house_id in house_ids:
    y_true, y_pred_sarima, y_pred_flft = load_pair(house_id, "test")
    y_pred_ens = best_alpha * y_pred_sarima + (1.0 - best_alpha) * y_pred_flft

    metrics = evaluate_raw_predictions_multistep(y_true, y_pred_ens)

    test_rows.append({
        "house_id": house_id,
        **metrics
    })

test_df = pd.DataFrame(test_rows)
test_df.to_csv(OUT_TEST_PER_HOUSE, index=False)

summary_df = pd.DataFrame([{
    "model": "blind_unseen_house_weighted_ensemble",
    "alpha_sarima": best_alpha,
    "alpha_flft": 1.0 - best_alpha,
    "mean_rmse_across_horizons": float(test_df["mean_rmse_across_horizons"].mean()),
    "median_rmse_across_horizons": float(test_df["mean_rmse_across_horizons"].median()),
    "mean_mae_across_horizons": float(test_df["mean_mae_across_horizons"].mean()),
    "median_mae_across_horizons": float(test_df["mean_mae_across_horizons"].median()),
    "mean_rmse_t+1": float(test_df["rmse_t+1"].mean()),
    "mean_rmse_t+2": float(test_df["rmse_t+2"].mean()),
    "mean_rmse_t+3": float(test_df["rmse_t+3"].mean()),
    "mean_rmse_t+4": float(test_df["rmse_t+4"].mean()),
    "mean_rmse_t+5": float(test_df["rmse_t+5"].mean()),
    "mean_rmse_t+6": float(test_df["rmse_t+6"].mean()),
    "mean_mae_t+1": float(test_df["mae_t+1"].mean()),
    "mean_mae_t+2": float(test_df["mae_t+2"].mean()),
    "mean_mae_t+3": float(test_df["mae_t+3"].mean()),
    "mean_mae_t+4": float(test_df["mae_t+4"].mean()),
    "mean_mae_t+5": float(test_df["mae_t+5"].mean()),
    "mean_mae_t+6": float(test_df["mae_t+6"].mean()),
    "n_houses": int(test_df["house_id"].nunique()),
}])

summary_df.to_csv(OUT_TEST_SUMMARY, index=False)

print("\nTest summary:")
print(summary_df)
print(f"\nSaved alpha search to: {OUT_ALPHA_SEARCH}")
print(f"Saved per-house test results to: {OUT_TEST_PER_HOUSE}")
print(f"Saved test summary to: {OUT_TEST_SUMMARY}")