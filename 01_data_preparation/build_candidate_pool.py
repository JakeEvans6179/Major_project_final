from pathlib import Path
import pandas as pd
import numpy as np

'''
**Step 1**

Script for accessing household data and filtering
Filter out ToU and only use standard tarrif data

Remove duplicate readings, calculate total duration of each house and compute coverage ratio (ratio of valid readings/ total expected readings)

Colect all eligible houses for plotting
'''
parquet_file = Path("saved_householddata.parquet")
WINDOW_DURATION = 789       #01-01-2012 --> 28-02-2014


# IF STATEMENT: Check if the file already exists
if parquet_file.exists():
    print(f"Found {parquet_file.name} Loading data directly")
    # Instantly load the compiled data
    all_df = pd.read_parquet(parquet_file)

else:
    print(f"{parquet_file.name} not found. Processing raw CSVs...")

    # folder containing all partitioned CSVs
    data_folder = Path(r"../data/Partitioned LCL Data")

    # find all csv files recursively
    csv_files = sorted(data_folder.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    dfs = []

    for i, f in enumerate(csv_files, start=1):
        print(f"Loading {i}/{len(csv_files)}: {f.name}")

        temp = pd.read_csv(
            f,
            usecols=["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour) "],
            low_memory=False
        )

        temp = temp.rename(columns={"KWH/hh (per half hour) ": "kwh"})

        # convert types
        temp["DateTime"] = pd.to_datetime(temp["DateTime"], errors="coerce")
        temp["kwh"] = pd.to_numeric(temp["kwh"], errors="coerce")

        # keep only standard tariff houses
        temp = temp[temp["stdorToU"] == "Std"].copy()

        dfs.append(temp)

    # combine into one dataframe
    all_df = pd.concat(dfs, ignore_index=True)

    # remove exact duplicates
    all_df = all_df.drop_duplicates(subset=["LCLid", "stdorToU", "DateTime", "kwh"])

    # drop std rating

    all_df = all_df.drop(columns=["stdorToU"])

    # Save to parquet file so the IF statement catches it next time
    all_df.to_parquet(parquet_file, index=False)
    print("Finished processing and saved to Parquet.")


# --- Regardless of how it was loaded, your data is now ready here ---
print("\n--- Data Summary ---")
print(all_df.head())
print("Shape:", all_df.shape)
print("Number of unique houses:", all_df["LCLid"].nunique())

print("Extracted house data")
print(all_df)

#convert kwh to numerical values
all_df['kwh'] = pd.to_numeric(all_df['kwh'], errors='coerce')


#testing = all_df[all_df['LCLid'] == 'MAC000022']
#print(testing)
# convert kwh to numerical values

'''
Remove rows with timestamps not in expected format xx:00:00 xx:30:00
'''
print("Removing off-grid timestamps globally")

on_grid_mask = (
    all_df["DateTime"].dt.minute.isin([0, 30]) &
    (all_df["DateTime"].dt.second == 0)
)

off_grid_rows = all_df[~on_grid_mask].copy()

print("Off-grid rows found:", len(off_grid_rows))
print("Affected houses:", off_grid_rows["LCLid"].nunique())
print(off_grid_rows.head(20))

# drop all off-grid rows globally
all_df = all_df[on_grid_mask].copy()

print("Rows after dropping off-grid rows:", len(all_df))
print("Unique houses after dropping off-grid rows:", all_df["LCLid"].nunique())


print("Sorting household data by houseid and DateTime")
#sort by household id, then by dateTime
#Reset row number at each new household
all_df = all_df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

print("Sorted house data --> house id --> DataTime")
print(all_df)


print("Calculating household stats (Duration, coverage)")
household_stats = (
    all_df.groupby("LCLid") #takes dataset and splits into smaller chunks (one per house)
    .agg(       #extracts summary statistics for each house
        First_timestep=("DateTime", "min"), #gets first (min) datetime house was recorded
        Last_timestep=("DateTime", "max"),
        Valid_count=("kwh", lambda x: x.notna().sum())      #counts number of rows with actual numbers (ignore NaN/ missing values)
    ).reset_index().rename(columns={"LCLid": "Household_id"})   #rename column to Household_id
)


#print(household_stats)

household_stats["Total_count"] = (
    ((household_stats["Last_timestep"] - household_stats["First_timestep"]).dt.total_seconds() / (30 * 60)) #How many readings house should have in the interval
    .round()
    .astype(int)
    + 1 #Add one to account for initial starting point
)

household_stats["Coverage"] = household_stats["Valid_count"] / household_stats["Total_count"]   #find ratio of readings with data to total readings expected in period

# record length in days
household_stats["Span_days"] = (
    (household_stats["Last_timestep"] - household_stats["First_timestep"]).dt.total_seconds()
    / (24 * 60 * 60)
)

print("Household comparison stats:")
print(household_stats) #all house metrics are calculated


'''
New
'''

#Find houses within a fixed date window allowing for direct comparison


print(household_stats["Last_timestep"].value_counts().head(20)) #see most common last timesteps

#print("something")
#wait = input("Press Enter to continue.")
#print("something")

common_end = pd.Timestamp("2014-02-28 00:00:00")
common_start = common_end - pd.Timedelta(days=WINDOW_DURATION)
tolerance = pd.Timedelta(minutes=30)

print("Common start:", common_start)
print("Common end:  ", common_end)

print("Testing to see how many houses end at or after 2014-02-28 00:00:00")
valid_houses = household_stats[
    (household_stats["First_timestep"] <= common_start + tolerance) & (household_stats["Last_timestep"] >= common_end - tolerance)].copy()

print("Houses fully covering the fixed window:", len(valid_houses))
pd.set_option('display.max_columns', None)
print(valid_houses)


eligible_ids = valid_houses["Household_id"].tolist()        #save house ids to list to filter for the eligible houses later

window_df = all_df[
    (all_df["DateTime"] >= common_start) &      #only include houses from eligible list, starting and stopping from a fixed time
    (all_df["DateTime"] <= common_end) &
    (all_df["LCLid"].isin(eligible_ids))
].copy()

#window_df contains all households that fit length criteria and start and stop at the same time

print(window_df)

#Find households within the filters (>800 days duration, >0.99 coverage rating)
#good_houses = household_stats[(household_stats['Coverage'] > 0.99) & (household_stats["Span_days"] > 800)]
expected_count = WINDOW_DURATION * 48 + 1

print("Eligible houses spanning full window:", len(valid_houses))       #counts rows after removing duplication, filtering and removing off grid value
print("Expected timestamps per house:", expected_count)         #counts total number of rows per house assuming perfect data
print("Expected total rows if complete:", expected_count * len(valid_houses))       #total number of rows assuming perfect data across all households
print("Actual rows in window_df:", len(window_df))
print("Missing rows vs perfect completeness:", expected_count * len(valid_houses) - len(window_df))
#print(good_houses)




print("Calculating coverage inside window_df")

#Recalculate coverage inside window_df

window_stats = (
    window_df.groupby("LCLid")
    .agg(
        Valid_count=("kwh", lambda x: x.notna().sum()),
        Unique_timestamps=("DateTime", "nunique")
    )
    .reset_index()
    .rename(columns={"LCLid": "Household_id"})
)

window_stats["Total_count"] = expected_count
window_stats["Coverage"] = window_stats["Valid_count"] / window_stats["Total_count"]
window_stats["Timestamp_coverage"] = window_stats["Unique_timestamps"] / window_stats["Total_count"]

print("\nFixed-window household stats:")
print(window_stats.head())
print(window_stats["Coverage"].describe())

#final quality filter
good_houses = window_stats[window_stats["Coverage"] > 0.99].copy()

print("\nGood houses after fixed-window filter:", len(good_houses))
print(good_houses.head())

# ---------------------------------------------------
# Randomly sample 100 UNIQUE household IDs
# ---------------------------------------------------
print("\nRandomly selecting household IDs from eligible list")
rng = np.random.default_rng(6769)

print("Eligible houses:", len(good_houses))

if len(good_houses) < 100:
    raise ValueError("Too few eligible houses to sample")
'''
selected_ids = rng.choice(
    good_houses["Household_id"].to_numpy(),
    size=100,
    replace=False
)
'''
selected_ids = good_houses["Household_id"].to_numpy()   #take all eligible houses 
selected_houses_lst = pd.DataFrame({"Household_id": selected_ids})

print(selected_houses_lst)
print("Selected houses:", len(selected_houses_lst))

# ---------------------------------------------------
# Filter raw fixed-window data to houses
# ---------------------------------------------------
selected_houses = window_df[window_df["LCLid"].isin(selected_ids)].copy()

print("\nSelected raw rows:")
print(selected_houses)
print("Unique selected houses:", selected_houses["LCLid"].nunique())
print("Selected raw shape:", selected_houses.shape)
'''
# Optional save
selected_100.to_csv("selected_100_houses_fixed800d.csv", index=False)
selected_houses.to_parquet("selected_100_households_raw_fixed800d.parquet", index=False)
good_houses.to_csv("good_houses_fixed800d.csv", index=False)
window_stats.to_csv("window_stats_fixed800d.csv", index=False)

'''
#save to parquet for training script to use
selected_houses.to_parquet("eligible_households_raw.parquet", index=False)







