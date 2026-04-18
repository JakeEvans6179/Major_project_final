from pathlib import Path
import pandas as pd
import Helper_functions

"""
Build one 24-hour average load profile for each house
using TRAIN data only.

Output:
- one row = one house
- columns h00 to h23 = average kwh at each hour of day
"""

DATA_PATH = Path("../data_files/final_locked_100_normalised.parquet")
MAX_MIN_PATH = Path("../data_files/global_weather_scaler.csv")
LOCAL_KWH_SCALING = Path("../data_files/local_kwh_scaler.csv")

# Load full dataset
df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = (
    Helper_functions.load_data(DATA_PATH, MAX_MIN_PATH, LOCAL_KWH_SCALING)
)

# Get all unique house IDs
house_ids = sorted(df["LCLid"].unique())

# We will store one dictionary per house here
profile_rows = []

for house_id in house_ids:
    print(f"Processing {house_id}")

    # Keep only this house and only the TRAIN split
    house_train = df[
        (df["LCLid"] == house_id) &
        (df["split"] == "train")
    ].copy()

    # Skip if no train data
    if house_train.empty:
        print(f"Skipping {house_id}: empty train split")
        continue

    # Extract hour of day from DateTime
    # Example: 2023-01-01 13:00:00 -> 13
    house_train["hour"] = house_train["DateTime"].dt.hour

    # For each hour 0-23, calculate the average kwh
    hourly_mean = house_train.groupby("hour")["kwh"].mean()

    # Make sure all 24 hours exist and are in the right order
    hourly_mean = hourly_mean.reindex(range(24))

    # Create one row for this house
    row = {"house_id": house_id}

    # Save each hour's average into columns h00, h01, ..., h23
    for h in range(24):
        row[f"h{h:02d}"] = hourly_mean.loc[h]

    # Add this house row to the list
    profile_rows.append(row)

# Turn list of rows into a dataframe
hourly_profiles = pd.DataFrame(profile_rows)

print(hourly_profiles.shape)
print(hourly_profiles.head())

# Save to csv

print(hourly_profiles.head())
#hourly_profiles.to_csv("household_24h_train_profiles.csv", index=False)

cluster_features = hourly_profiles.drop(columns=["house_id"])
print(cluster_features.head())