from pathlib import Path
import numpy as np
import pandas as pd

"""
Create a limited-data unseen-house dataset using one fixed 14-day window:
- 5 days train
- 2 days val
- 7 days test

Scaling:
- per-house min-max for kwh using train only
- global min-max for weather using all sampled houses' train only
"""

hourly_path = Path("unseen_household_weather_merged.parquet")

out_parquet = Path("unseen_2week_normalised.parquet")
out_weather_scaler = Path("unseen_global_weather_scaler.csv")
out_kwh_scaler = Path("unseen_local_kwh_scaler.csv")

df = pd.read_parquet(hourly_path)
df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
df = df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

# --------------------------------------------------
# FIXED 14-DAY WINDOW
# Choose ONE of these
# --------------------------------------------------

# Option A: first two weeks of 2012
WINDOW_START = pd.Timestamp("2012-01-01 00:00:00")
WINDOW_END   = pd.Timestamp("2012-01-15 00:00:00")   # exclusive

# Option B: if you really want to start later, uncomment this instead
# WINDOW_START = pd.Timestamp("2012-01-10 00:00:00")
# WINDOW_END   = pd.Timestamp("2012-01-24 00:00:00")   # exclusive

window_df = df[
    (df["DateTime"] >= WINDOW_START) &
    (df["DateTime"] < WINDOW_END)
].copy()

print("Window start:", WINDOW_START)
print("Window end:", WINDOW_END)
print("Windowed shape:", window_df.shape)
print("Unique houses:", window_df["LCLid"].nunique())

# --------------------------------------------------
# SPLIT CONFIG
# --------------------------------------------------
TRAIN_DAYS = 5
VAL_DAYS = 2
TEST_DAYS = 7

assert TRAIN_DAYS + VAL_DAYS + TEST_DAYS == 14

HOURS_PER_DAY = 24
TRAIN_LEN = TRAIN_DAYS * HOURS_PER_DAY
VAL_LEN = VAL_DAYS * HOURS_PER_DAY
TEST_LEN = TEST_DAYS * HOURS_PER_DAY

print("Train hours:", TRAIN_LEN)
print("Val hours:", VAL_LEN)
print("Test hours:", TEST_LEN)

# --------------------------------------------------
# SPLIT + SCALERS
# --------------------------------------------------
house_splits = {}
house_kwh_scalers = []
all_train_temp = []
all_train_hum = []

house_ids = sorted(window_df["LCLid"].unique())

for house_id in house_ids:
    house = window_df[window_df["LCLid"] == house_id].copy().sort_values("DateTime")
    n = len(house)

    expected_n = TRAIN_LEN + VAL_LEN + TEST_LEN
    assert n == expected_n, f"{house_id}: expected {expected_n} rows, got {n}"

    train_end = TRAIN_LEN
    val_end = TRAIN_LEN + VAL_LEN

    train_df = house.iloc[:train_end].copy()
    val_df = house.iloc[train_end:val_end].copy()
    test_df = house.iloc[val_end:].copy()

    kwh_min = train_df["kwh"].min()
    kwh_max = train_df["kwh"].max()

    if kwh_max == kwh_min:
        raise ValueError(f"{house_id}: kwh_min and kwh_max are equal")

    house_splits[house_id] = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    house_kwh_scalers.append({
        "house_id": house_id,
        "kwh_min": kwh_min,
        "kwh_max": kwh_max
    })

    all_train_temp.append(train_df["temperature"])
    all_train_hum.append(train_df["humidity"])

all_train_temp = pd.concat(all_train_temp, ignore_index=True)
all_train_hum = pd.concat(all_train_hum, ignore_index=True)
house_kwh_scalers = pd.DataFrame(house_kwh_scalers)

global_temp_min = all_train_temp.min()
global_temp_max = all_train_temp.max()
global_hum_min = all_train_hum.min()
global_hum_max = all_train_hum.max()

print("\nGlobal training temperature min:", global_temp_min)
print("Global training temperature max:", global_temp_max)
print("Global training humidity min:", global_hum_min)
print("Global training humidity max:", global_hum_max)

if global_temp_max == global_temp_min:
    raise ValueError("Global temperature min and max are equal")

if global_hum_max == global_hum_min:
    raise ValueError("Global humidity min and max are equal")

# --------------------------------------------------
# MIN-MAX SCALE
# --------------------------------------------------
def minmax_scale(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)

for house_id in house_ids:
    house_kwh_min = house_kwh_scalers.loc[
        house_kwh_scalers["house_id"] == house_id, "kwh_min"
    ].item()
    house_kwh_max = house_kwh_scalers.loc[
        house_kwh_scalers["house_id"] == house_id, "kwh_max"
    ].item()

    # kwh: local per-house scaler from train only
    for split in ["train", "val", "test"]:
        house_splits[house_id][split]["kwh"] = minmax_scale(
            house_splits[house_id][split]["kwh"],
            house_kwh_min,
            house_kwh_max
        )

    # weather: global scaler from all train rows only
    for split in ["train", "val", "test"]:
        house_splits[house_id][split]["temperature"] = minmax_scale(
            house_splits[house_id][split]["temperature"],
            global_temp_min,
            global_temp_max
        )
        house_splits[house_id][split]["humidity"] = minmax_scale(
            house_splits[house_id][split]["humidity"],
            global_hum_min,
            global_hum_max
        )

# --------------------------------------------------
# COMBINE + SAVE
# --------------------------------------------------
rows = []

for house_id in house_ids:
    train_df = house_splits[house_id]["train"].copy()
    train_df["split"] = "train"
    rows.append(train_df)

    val_df = house_splits[house_id]["val"].copy()
    val_df["split"] = "val"
    rows.append(val_df)

    test_df = house_splits[house_id]["test"].copy()
    test_df["split"] = "test"
    rows.append(test_df)

splits_df = pd.concat(rows, ignore_index=True)

print("\nFinal split dataframe shape:", splits_df.shape)
print(splits_df["split"].value_counts())
print("Unique houses:", splits_df["LCLid"].nunique())

print("\nScaled train temperature range:")
print(
    splits_df.loc[splits_df["split"] == "train", "temperature"].min(),
    splits_df.loc[splits_df["split"] == "train", "temperature"].max()
)

print("\nScaled train humidity range:")
print(
    splits_df.loc[splits_df["split"] == "train", "humidity"].min(),
    splits_df.loc[splits_df["split"] == "train", "humidity"].max()
)

print("\nScaled train kwh range:")
print(
    splits_df.loc[splits_df["split"] == "train", "kwh"].min(),
    splits_df.loc[splits_df["split"] == "train", "kwh"].max()
)

splits_df.to_parquet(out_parquet, index=False)

pd.DataFrame({
    "global_temp_min": [global_temp_min],
    "global_temp_max": [global_temp_max],
    "global_hum_min": [global_hum_min],
    "global_hum_max": [global_hum_max],
}).to_csv(out_weather_scaler, index=False)

house_kwh_scalers.to_csv(out_kwh_scaler, index=False)

print(f"\nSaved to: {out_parquet}")
print(f"Saved weather scalers to: {out_weather_scaler}")
print(f"Saved local kwh scalers to: {out_kwh_scaler}")