import pandas as pd
from pathlib import Path

'''
**Step 2**
Extract weather data and clean/ apply forward filling
'''
# folder containing all partitioned CSVs
data_folder = Path("../data/weather_data")
WINDOW_DURATION = 789

out_parquet = Path("weather_data.parquet")

# find all csv files recursively
csv_files = sorted(data_folder.rglob("*.csv"))
print(f"Found {len(csv_files)} CSV files")

dfs = []

for i, f in enumerate(csv_files, start=1):
    print(f"Loading {i}/{len(csv_files)}: {f.name}")

    temp = pd.read_csv(
        f,
        usecols=["ob_time", "air_temperature", "rltv_hum"],
        low_memory=False,
        skiprows=283
    )

    temp = temp.rename(columns={"air_temperature": "temperature", "rltv_hum": "humidity", "ob_time": "DateTime"})

    # convert types
    temp["DateTime"] = pd.to_datetime(temp["DateTime"], errors="coerce")  #make the format same as energy dataset
    temp["temperature"] = pd.to_numeric(temp["temperature"], errors="coerce")
    temp["humidity"] = pd.to_numeric(temp["humidity"], errors="coerce")

    print(temp.tail())
    dfs.append(temp)

# combine into one dataframe
all_weather = pd.concat(dfs, ignore_index=True)


print("Rows before dropping NaT:", len(all_weather))
print("NaT rows:", all_weather["DateTime"].isna().sum())

all_weather = all_weather.dropna(subset=["DateTime"]).copy()    #drop any rows with invalid time

print("Rows after dropping NaT:", len(all_weather))


all_weather = all_weather.drop_duplicates(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)   #remove duplicates


'''
Remove rows with timestamps not in hourly format
'''
print("Removing off-grid timestamps globally")

on_grid_mask = (
    all_weather["DateTime"].dt.minute.isin([0]) &
    (all_weather["DateTime"].dt.second == 0)
)

off_grid_rows = all_weather[~on_grid_mask].copy()

print("Off-grid rows found:", len(off_grid_rows))
print(off_grid_rows.head(20))

# drop all off-grid rows globally
all_weather = all_weather[on_grid_mask].copy()

print("Rows after dropping off-grid rows:", len(all_weather))

print(all_weather)

#compute coverage for visualisation
common_end = pd.Timestamp("2014-02-28 00:00:00")
common_start = pd.Timestamp("2012-01-01 00:00:00")

#apply windowing to get the data for the respective dates
window_weather = all_weather[(all_weather["DateTime"] >= common_start) & (all_weather["DateTime"] < common_end)].copy() #end at 2014-02-27 23:00:00 as that is when demand data ends after resampling to hourly

print(window_weather)

#compute coverage for both temperature and humidity
#calculate the expected number of timesteps
print(window_weather.head())
print(window_weather.tail())
print("Rows in window:", len(window_weather))

#expected hourly timestamps in window matching demand data
expected_timesteps = int((common_end - common_start).total_seconds() / 3600)
print("Expected hourly timestamps:", expected_timesteps)

#calculate coverage stats
temperature_valid = window_weather["temperature"].notna().sum()
humidity_valid = window_weather["humidity"].notna().sum()
unique_timestamps = window_weather["DateTime"].nunique()

window_stats = pd.DataFrame([{
    "temperature_valid": temperature_valid,
    "humidity_valid": humidity_valid,
    "unique_timestamps": unique_timestamps,
    "total_count": expected_timesteps,
    "temperature_coverage": temperature_valid / expected_timesteps,
    "humidity_coverage": humidity_valid / expected_timesteps,
    "timestamp_coverage": unique_timestamps / expected_timesteps
}])

pd.set_option('display.max_columns', None)
print("\nWeather coverage stats:")
print(window_stats) #current dataset --> 1 missing timestamp, 3 missing temp and 3 missing humidity values
#add in missing timestamp and apply forward filling for temp and humidity


expected_index = pd.date_range(start=common_start, end=common_end - pd.Timedelta(hours=1), freq="h")    #remove last hour as demand data only goes up to 2014-02-27 23:00:00 after resampling

missing_times = expected_index.difference(window_weather["DateTime"])

print("Number of missing timestamps:", len(missing_times))
print(missing_times)

#add in the missing timestep
weather_full = (
    window_weather
    .set_index("DateTime")
    .reindex(expected_index)
    .rename_axis("DateTime")
    .reset_index()
)


print("Missing temperature after reindex:", weather_full["temperature"].isna().sum())
print("Missing humidity after reindex:", weather_full["humidity"].isna().sum())

print(weather_full[weather_full["temperature"].isna() | weather_full["humidity"].isna()])   #show the rows with either missing temp or humidity

#apply forward filling
weather_full["temperature"] = weather_full["temperature"].ffill().bfill()
weather_full["humidity"] = weather_full["humidity"].ffill().bfill()

#check there are no rows with missing data
print("Remaining missing temperature:", weather_full["temperature"].isna().sum())
print("Remaining missing humidity:", weather_full["humidity"].isna().sum())
print(weather_full.head())
print(weather_full.tail())

#raise error if there are still rows with NaN
assert len(weather_full) == expected_timesteps
assert weather_full["temperature"].isna().sum() == 0
assert weather_full["humidity"].isna().sum() == 0

weather_full.to_parquet(out_parquet, index=False)
print(f"\nSaved processed hourly data to: {out_parquet}")

print("Final row count:", len(weather_full))
print("Expected row count:", expected_timesteps)




