from pathlib import Path
import numpy as np
import pandas as pd

"""
**step 3**
Preprocess the 10 unseen households:

- load fixed-window raw rows
- reindex each house to the expected 30-minute timeline
- forward fill missing kwh
- backfill only if the first value is missing
- shift timestamps back 30 mins
- aggregate to hourly kwh
- add time features
- combine all houses into one dataframe
- save to parquet

**updated
Load in the processed weather data and merge into dataset in this script
Normalise in next script
"""

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
raw_parquet = Path("unseen_house_sample.parquet")
out_parquet = Path("unseen_household_weather_merged.parquet")

WINDOW_DURATION = 789

common_end = pd.Timestamp("2014-02-28 00:00:00")
common_start = common_end - pd.Timedelta(days=WINDOW_DURATION)

print("common start:", common_start)

# exact half-hour schedule expected for every house
expected_index_30m = pd.date_range(
    start=common_start + pd.Timedelta(minutes=30),      #start at the first half hour after starting, removes the first half hour reading
    end=common_end,
    freq="30min"
)

print("Expected 30-min points per house:", len(expected_index_30m))

# --------------------------------------------------
# LOAD
# --------------------------------------------------
df = pd.read_parquet(raw_parquet)

df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")

df = df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

print(df.head())
print("Unique houses:", df["LCLid"].nunique())
print("Raw shape:", df.shape)

# --------------------------------------------------
# HOUSE PREPROCESS FUNCTION
# --------------------------------------------------
def preprocess_one_house(house_df, house_id):
    house = house_df.copy()

    # set index to time
    house = house.set_index("DateTime").sort_index()    #set index to DateTime

    #keep only kwh column during reindex
    house = house[["kwh"]]

    #add missing timestamps as NaN rows
    house = house.reindex(expected_index_30m)

    #fill missing consumption values
    #ffill handles normal internal gaps
    #bfill only fixes a missing first boundary value if needed
    house["kwh"] = house["kwh"].ffill().bfill()

    #restore household ID
    house["LCLid"] = house_id

    #shift timestamps back 30 minutes so hourly aggregation sums correctly

    house.index = house.index - pd.Timedelta(minutes=30)

    #aggregate to hourly kwh
    hourly = (
        house[["kwh"]]
        .resample("h")
        .sum()
    )

    #put house id back in
    hourly["LCLid"] = house_id

    #add cyclical/calendar features
    seconds = hourly.index.map(pd.Timestamp.timestamp)

    day_duration = 24 * 60 * 60
    year_duration = 365.2425 * day_duration

    hourly["hour_sin"] = np.sin(2 * np.pi * seconds / day_duration)
    hourly["hour_cos"] = np.cos(2 * np.pi * seconds / day_duration)

    hourly["year_sin"] = np.sin(2 * np.pi * seconds / year_duration)
    hourly["year_cos"] = np.cos(2 * np.pi * seconds / year_duration)

    dow = hourly.index.dayofweek
    hourly["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    hourly["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    hourly["weekend"] = (hourly.index.dayofweek >= 5).astype(int)

    #return datetime to a column
    hourly = hourly.reset_index().rename(columns={"index": "DateTime"})

    return hourly


# --------------------------------------------------
# RUN FOR ALL HOUSES
# --------------------------------------------------
processed = []

house_ids = sorted(df["LCLid"].unique())

for i, house_id in enumerate(house_ids, start=1):       #start process for all 100 houses
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    house_df = df[df["LCLid"] == house_id].copy()
    hourly_house = preprocess_one_house(house_df, house_id)
    processed.append(hourly_house)

hourly_df = pd.concat(processed, ignore_index=True)     #put all the processed dataframes into one big dataframe
#pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)
print("\nProcessed hourly dataframe:")
print(hourly_df)
print("Unique houses:", hourly_df["LCLid"].nunique())
print("Hourly shape:", hourly_df.shape)

#check remaining NaNs
print("\nRemaining NaNs by column:")
print(hourly_df.isna().sum())



'''
Merge dataset with weather data, one copy for each houseid
'''

weather_parquet = Path("weather_data.parquet")
weather_df = pd.read_parquet(weather_parquet)

weather_df["DateTime"] = pd.to_datetime(weather_df["DateTime"], errors="coerce")

print("\nWeather dataframe:")
print(weather_df.head())
print("Weather shape:", weather_df.shape)
print("Weather duplicate timestamps:", weather_df.duplicated(subset=["DateTime"]).sum())

hourly_df = hourly_df.merge(
    weather_df,
    on="DateTime",
    how="left"
)

print("\nAfter weather merge:")
print(hourly_df.head())
print("Merged shape:", hourly_df.shape)
print("Duplicate LCLid-DateTime rows:", hourly_df.duplicated(subset=["LCLid", "DateTime"]).sum())
print("Missing temperature after merge:", hourly_df["temperature"].isna().sum())
print("Missing humidity after merge:", hourly_df["humidity"].isna().sum())

#save
hourly_df.to_parquet(out_parquet, index=False)
print(f"\nSaved processed hourly data to: {out_parquet}")


