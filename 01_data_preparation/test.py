from pathlib import Path
import pandas as pd
import numpy as np

'''
**Step 1**

Script for accessing household data and filtering
Filter out ToU and only use standard tariff data

Remove duplicate readings, calculate total duration of each house and compute coverage ratio
(ratio of valid readings / total expected readings)

Collect all eligible houses for plotting
'''

# Define the path for your saved Parquet file
parquet_file = Path("saved_householddata.parquet")

WINDOW_DURATION = 789       # 01-01-2012 --> 28-02-2014

# Set this to True if you want to rerun the raw CSV processing and regenerate eLogbook counts
FORCE_REPROCESS = True

# eLogbook count storage
elogbook_counts = {}


def log_count(name, value):
    """Print and store an eLogbook count."""
    elogbook_counts[name] = value
    print(f"[ELOGBOOK] {name}: {value}")


# IF STATEMENT: Check if the file already exists
if parquet_file.exists() and not FORCE_REPROCESS:
    print(f"Found {parquet_file.name}. Loading data directly")

    # Instantly load the compiled data
    all_df = pd.read_parquet(parquet_file)

else:
    print(f"{parquet_file.name} not found or FORCE_REPROCESS=True. Processing raw CSVs...")

    # folder containing all partitioned CSVs
    data_folder = Path(r"../data/Partitioned LCL Data")

    # find all csv files recursively
    csv_files = sorted(data_folder.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    dfs = []

    # ---------------------------------------------------
    # eLogbook counters for raw CSV loading
    # ---------------------------------------------------
    raw_row_count = 0
    raw_house_ids = set()
    std_house_ids = set()
    non_std_house_ids = set()

    raw_invalid_timestamp_count = 0
    raw_missing_kwh_count = 0

    std_row_count_before_duplicate_removal = 0
    std_invalid_timestamp_count = 0
    std_missing_kwh_count = 0

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

        # ---------------------------------------------------
        # eLogbook counting only: raw data before tariff filter
        # ---------------------------------------------------
        raw_row_count += len(temp)

        raw_house_ids.update(
            temp["LCLid"].dropna().unique()
        )

        std_house_ids.update(
            temp.loc[temp["stdorToU"] == "Std", "LCLid"].dropna().unique()
        )

        non_std_house_ids.update(
            temp.loc[temp["stdorToU"] != "Std", "LCLid"].dropna().unique()
        )

        raw_invalid_timestamp_count += temp["DateTime"].isna().sum()
        raw_missing_kwh_count += temp["kwh"].isna().sum()

        # keep only standard tariff houses
        temp = temp[temp["stdorToU"] == "Std"].copy()

        # ---------------------------------------------------
        # eLogbook counting only: after tariff filter
        # ---------------------------------------------------
        std_row_count_before_duplicate_removal += len(temp)
        std_invalid_timestamp_count += temp["DateTime"].isna().sum()
        std_missing_kwh_count += temp["kwh"].isna().sum()

        dfs.append(temp)

    # ---------------------------------------------------
    # eLogbook printout: raw loading and tariff filtering
    # ---------------------------------------------------
    removed_house_ids = raw_house_ids - std_house_ids
    mixed_tariff_ids = std_house_ids & non_std_house_ids

    print("\n--- ELOGBOOK COUNTS: RAW CSV LOADING / TARIFF FILTERING ---")
    log_count("Raw CSV files loaded", len(csv_files))
    log_count("Raw demand rows before tariff filtering", raw_row_count)
    log_count("Raw unique households before tariff filtering", len(raw_house_ids))
    log_count("Number of standard tariff households retained", len(std_house_ids))
    log_count("Number of removed ToU / non-standard tariff households", len(removed_house_ids))
    log_count("Number of non-standard tariff households found", len(non_std_house_ids))
    log_count("Households with both Std and non-Std tariff labels", len(mixed_tariff_ids))
    log_count("Raw invalid demand timestamps before tariff filtering", raw_invalid_timestamp_count)
    log_count("Raw missing demand kWh values before tariff filtering", raw_missing_kwh_count)
    log_count("Standard tariff demand rows before duplicate removal", std_row_count_before_duplicate_removal)
    log_count("Standard tariff invalid demand timestamps before duplicate removal", std_invalid_timestamp_count)
    log_count("Standard tariff missing demand kWh values before duplicate removal", std_missing_kwh_count)
    print("----------------------------------------------------------\n")

    # combine into one dataframe
    all_df = pd.concat(dfs, ignore_index=True)

    # ---------------------------------------------------
    # eLogbook printout: duplicates before removal
    # ---------------------------------------------------
    duplicate_count = all_df.duplicated(
        subset=["LCLid", "stdorToU", "DateTime", "kwh"]
    ).sum()

    print("\n--- ELOGBOOK COUNTS: DEMAND DUPLICATES ---")
    log_count("Duplicate demand records removed", duplicate_count)
    log_count("Rows before duplicate removal", len(all_df))

    # remove exact duplicates
    all_df = all_df.drop_duplicates(subset=["LCLid", "stdorToU", "DateTime", "kwh"])

    log_count("Rows after duplicate removal", len(all_df))
    print("----------------------------------------------------------\n")

    # drop std rating
    all_df = all_df.drop(columns=["stdorToU"])

    print("\n--- ELOGBOOK COUNTS: POST-TARIFF CLEANED DEMAND DATA ---")
    log_count("Rows after tariff filtering and duplicate removal", len(all_df))
    log_count("Unique standard tariff households after duplicate removal", all_df["LCLid"].nunique())
    print("----------------------------------------------------------\n")

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

# convert kwh to numerical values
all_df["kwh"] = pd.to_numeric(all_df["kwh"], errors="coerce")


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

log_count("Off-grid demand timestamp rows removed", len(off_grid_rows))
log_count("Households affected by off-grid demand timestamps", off_grid_rows["LCLid"].nunique())

# drop all off-grid rows globally
all_df = all_df[on_grid_mask].copy()

print("Rows after dropping off-grid rows:", len(all_df))
print("Unique houses after dropping off-grid rows:", all_df["LCLid"].nunique())

log_count("Rows after dropping off-grid demand timestamps", len(all_df))
log_count("Unique households after dropping off-grid demand timestamps", all_df["LCLid"].nunique())


print("Sorting household data by houseid and DateTime")

# sort by household id, then by DateTime
# reset row number at each new household
all_df = all_df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

print("Sorted house data --> house id --> DateTime")
print(all_df)


print("Calculating household stats (Duration, coverage)")

household_stats = (
    all_df.groupby("LCLid")
    .agg(
        First_timestep=("DateTime", "min"),
        Last_timestep=("DateTime", "max"),
        Valid_count=("kwh", lambda x: x.notna().sum())
    )
    .reset_index()
    .rename(columns={"LCLid": "Household_id"})
)

household_stats["Total_count"] = (
    (
        (household_stats["Last_timestep"] - household_stats["First_timestep"])
        .dt.total_seconds() / (30 * 60)
    )
    .round()
    .astype(int)
    + 1
)

household_stats["Coverage"] = household_stats["Valid_count"] / household_stats["Total_count"]

# record length in days
household_stats["Span_days"] = (
    (household_stats["Last_timestep"] - household_stats["First_timestep"]).dt.total_seconds()
    / (24 * 60 * 60)
)

print("Household comparison stats:")
print(household_stats)

log_count("Households after tariff, duplicate, and off-grid cleaning", household_stats["Household_id"].nunique())
log_count("Earliest household first timestep", household_stats["First_timestep"].min())
log_count("Latest household last timestep", household_stats["Last_timestep"].max())


'''
New
'''

# Find houses within a fixed date window allowing for direct comparison

print(household_stats["Last_timestep"].value_counts().head(20))

common_end = pd.Timestamp("2014-02-28 00:00:00")
common_start = common_end - pd.Timedelta(days=WINDOW_DURATION)
tolerance = pd.Timedelta(minutes=30)

print("Common start:", common_start)
print("Common end:  ", common_end)

log_count("Fixed study window start", common_start)
log_count("Fixed study window end", common_end)
log_count("Fixed study window duration days", WINDOW_DURATION)

print("Testing to see how many houses end at or after 2014-02-28 00:00:00")

valid_houses = household_stats[
    (household_stats["First_timestep"] <= common_start + tolerance) &
    (household_stats["Last_timestep"] >= common_end - tolerance)
].copy()

print("Houses fully covering the fixed window:", len(valid_houses))
pd.set_option("display.max_columns", None)
print(valid_houses)

log_count("Households fully covering fixed study window", len(valid_houses))


eligible_ids = valid_houses["Household_id"].tolist()

window_df = all_df[
    (all_df["DateTime"] >= common_start) &
    (all_df["DateTime"] <= common_end) &
    (all_df["LCLid"].isin(eligible_ids))
].copy()

# window_df contains all households that fit length criteria and start and stop at the same time

print(window_df)

expected_count = WINDOW_DURATION * 48 + 1

print("Eligible houses spanning full window:", len(valid_houses))
print("Expected timestamps per house:", expected_count)
print("Expected total rows if complete:", expected_count * len(valid_houses))
print("Actual rows in window_df:", len(window_df))
print("Missing rows vs perfect completeness:", expected_count * len(valid_houses) - len(window_df))

log_count("Expected half-hourly timestamps per household in fixed window", expected_count)
log_count("Expected total half-hourly rows if complete", expected_count * len(valid_houses))
log_count("Actual fixed-window half-hourly rows", len(window_df))
log_count(
    "Missing fixed-window rows versus perfect completeness",
    expected_count * len(valid_houses) - len(window_df)
)


print("Calculating coverage inside window_df")

# Recalculate coverage inside window_df

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

log_count("Minimum household coverage in fixed window", window_stats["Coverage"].min())
log_count("Mean household coverage in fixed window", window_stats["Coverage"].mean())
log_count("Maximum household coverage in fixed window", window_stats["Coverage"].max())
log_count("Minimum timestamp coverage in fixed window", window_stats["Timestamp_coverage"].min())
log_count("Mean timestamp coverage in fixed window", window_stats["Timestamp_coverage"].mean())
log_count("Maximum timestamp coverage in fixed window", window_stats["Timestamp_coverage"].max())

# final quality filter
good_houses = window_stats[window_stats["Coverage"] > 0.99].copy()

print("\nGood houses after fixed-window filter:", len(good_houses))
print(good_houses.head())

log_count("Households passing fixed-window coverage filter > 0.99", len(good_houses))
log_count("Households rejected by fixed-window coverage filter", len(window_stats) - len(good_houses))


# ---------------------------------------------------
# Randomly sample 100 UNIQUE household IDs
# ---------------------------------------------------
print("\nRandomly selecting household IDs from eligible list")
rng = np.random.default_rng(6769)

print("Eligible houses:", len(good_houses))
log_count("Eligible households before final selection", len(good_houses))
log_count("Random seed used for household selection", 6769)

if len(good_houses) < 100:
    raise ValueError("Too few eligible houses to sample")

'''
selected_ids = rng.choice(
    good_houses["Household_id"].to_numpy(),
    size=100,
    replace=False
)
'''

selected_ids = good_houses["Household_id"].to_numpy()   # take all eligible houses
selected_houses_lst = pd.DataFrame({"Household_id": selected_ids})

print(selected_houses_lst)
print("Selected houses:", len(selected_houses_lst))

log_count("Selected household count", len(selected_houses_lst))

# ---------------------------------------------------
# Filter raw fixed-window data to houses
# ---------------------------------------------------
selected_houses = window_df[window_df["LCLid"].isin(selected_ids)].copy()

print("\nSelected raw rows:")
print(selected_houses)
print("Unique selected houses:", selected_houses["LCLid"].nunique())
print("Selected raw shape:", selected_houses.shape)

log_count("Final selected unique households", selected_houses["LCLid"].nunique())
log_count("Final selected half-hourly rows", len(selected_houses))

'''
# Optional save
selected_100.to_csv("selected_100_houses_fixed800d.csv", index=False)
selected_houses.to_parquet("selected_100_households_raw_fixed800d.parquet", index=False)
good_houses.to_csv("good_houses_fixed800d.csv", index=False)
window_stats.to_csv("window_stats_fixed800d.csv", index=False)
'''

# save to parquet for training script to use
selected_houses.to_parquet("eligible_households_raw.parquet", index=False)


# ---------------------------------------------------
# Save final eLogbook counts
# ---------------------------------------------------
print("\n" + "=" * 80)
print("FINAL ELOGBOOK COUNTS")
print("=" * 80)

for key, value in elogbook_counts.items():
    print(f"{key}: {value}")

pd.DataFrame(
    list(elogbook_counts.items()),
    columns=["Metric", "Value"]
).to_csv("elogbook_counts.csv", index=False)

print("=" * 80)
print("Saved eLogbook counts to elogbook_counts.csv")