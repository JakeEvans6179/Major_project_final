import numpy as np
import pandas as pd
from pathlib import Path


'''
**step 4**
Get global and local household scalers
Same as global scalers, but min max kwh computed for each house and saved to csv for later processing

Returns data normalised parquet file (containing all household data + features + train, val, test split catagory)
Returns Global scaler csv for temp and humidity + local household scaler csv for demand

*these files can be used to train and run inference*
'''

hourly_path = Path("selected_100.parquet")

df = pd.read_parquet(hourly_path)



train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

assert abs(train_ratio + val_ratio + test_ratio -1) < 1e-9   #make sure it adds up to 100%


df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce") #set to DateTime object

#sort by houseid then by datetime within each houseid
df = df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

pd.set_option('display.max_columns', None)
print(df)
print("Unique houses:", df["LCLid"].nunique())
print("Shape:", df.shape)


#split data into train val test sets
house_splits = {}   #used to store each houses training, validation and test set
house_kwh_scalers = []  #used to collect all kwh data from training sets to find per house max and global min
all_train_temp = []
all_train_hum = []

house_ids = sorted(df["LCLid"].unique())        #sort house ids in ascending order

for house_id in house_ids:
    house = df[df["LCLid"] == house_id].copy().sort_values("DateTime")

    n = len(house)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = house.iloc[:train_end].copy()
    val_df = house.iloc[train_end:val_end].copy()
    test_df = house.iloc[val_end:].copy()

    #find min max values
    kwh_min = train_df["kwh"].min()
    kwh_max = train_df["kwh"].max()

    if kwh_max == kwh_min:
        raise ValueError(f"{house_id}: kwh_min and kwh_max cannot be equal")

    house_splits[house_id] = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    house_kwh_scalers.append({"house_id": house_id, "kwh_min": kwh_min, "kwh_max": kwh_max})
    all_train_temp.append(train_df["temperature"])
    all_train_hum.append(train_df["humidity"])


#combine all training kwh values across all houses --> used to calculate global min max values for scaling (on training set)
all_train_temp = pd.concat(all_train_temp, ignore_index=True)
all_train_hum = pd.concat(all_train_hum, ignore_index=True)

house_kwh_scalers = pd.DataFrame(house_kwh_scalers) #convert to dataframe

print(all_train_temp)
print(all_train_hum)

#get min and max values from training set
global_temp_min = all_train_temp.min()
global_temp_max = all_train_temp.max()

global_hum_min = all_train_hum.min()
global_hum_max = all_train_hum.max()



print("\nGlobal training temperature min:", global_temp_min)
print("Global training temperature max:", global_temp_max)

print("\nGlobal training humidity min:", global_hum_min)
print("Global training humidity max:", global_hum_max)



if global_temp_max == global_temp_min:
    raise ValueError("Global temperature min and max are equal; cannot apply min-max scaling.")

if global_hum_max == global_hum_min:
    raise ValueError("Global humidity min and max are equal; cannot apply min-max scaling.")
# --------------------------------------------------
# MIN-MAX SCALE KWH ONLY
# --------------------------------------------------
def minmax_scale(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)


#apply min max scaling for all sets (train, val, test)
for house_id in house_ids:
    #energy usage data
    house_kwh_min = house_kwh_scalers[house_kwh_scalers["house_id"] == house_id]["kwh_min"].item()
    house_kwh_max = house_kwh_scalers[house_kwh_scalers["house_id"] == house_id]["kwh_max"].item()

    house_splits[house_id]["train"]["kwh"] = minmax_scale(
        house_splits[house_id]["train"]["kwh"], house_kwh_min, house_kwh_max)

    house_splits[house_id]["val"]["kwh"] = minmax_scale(
        house_splits[house_id]["val"]["kwh"], house_kwh_min, house_kwh_max)

    house_splits[house_id]["test"]["kwh"] = minmax_scale(
        house_splits[house_id]["test"]["kwh"], house_kwh_min, house_kwh_max)

    #temperature data
    house_splits[house_id]["train"]["temperature"] = minmax_scale(
        house_splits[house_id]["train"]["temperature"], global_temp_min, global_temp_max)

    house_splits[house_id]["val"]["temperature"] = minmax_scale(
        house_splits[house_id]["val"]["temperature"], global_temp_min, global_temp_max)

    house_splits[house_id]["test"]["temperature"] = minmax_scale(
        house_splits[house_id]["test"]["temperature"], global_temp_min, global_temp_max)

    #humidity data
    house_splits[house_id]["train"]["humidity"] = minmax_scale(
        house_splits[house_id]["train"]["humidity"], global_hum_min, global_hum_max)

    house_splits[house_id]["val"]["humidity"] = minmax_scale(
        house_splits[house_id]["val"]["humidity"], global_hum_min, global_hum_max)

    house_splits[house_id]["test"]["humidity"] = minmax_scale(
        house_splits[house_id]["test"]["humidity"], global_hum_min, global_hum_max)


train_list = []
val_list = []
test_list = []

#Loop through every house and append its chunks to the correct list
#used later for centralised training
# --- For Later ---
for h in house_ids:
    train_list.append(house_splits[h]["train"])
    val_list.append(house_splits[h]["val"])
    test_list.append(house_splits[h]["test"])

#print(train_list)
#connect the populated lists together into one dataframe
train_all = pd.concat(train_list, ignore_index=True)
val_all = pd.concat(val_list, ignore_index=True)
test_all = pd.concat(test_list, ignore_index=True)

print("\nTrain shape:", train_all.shape)
print("Val shape:", val_all.shape)
print("Test shape:", test_all.shape)


print(train_all)

#sanity check, min = 0, max = 1
print("\nScaled train temperature range:")
print(train_all["temperature"].min(), train_all["temperature"].max())

print("\nScaled train humidity range:")
print(train_all["humidity"].min(), train_all["humidity"].max())

# ------

print("\nExample one house split sizes:")
example_house = house_ids[0]
print(example_house,
      len(house_splits[example_house]["train"]),
      len(house_splits[example_house]["val"]),
      len(house_splits[example_house]["test"]))



#create one df with all train, split, val data for all households to save and run in training model
#loop through each house, loop through training, val and test data, append to list
#once all houses appended, concatenate into dataframe and save to parquet file
rows = []

for house_id in house_ids:
    train_df = house_splits[house_id]["train"].copy() #get all training examples for house id
    train_df["split"] = "train"         #add label train to column called split
    rows.append(train_df)               #append to list

    val_df = house_splits[house_id]["val"].copy()
    val_df["split"] = "val"
    rows.append(val_df)

    test_df = house_splits[house_id]["test"].copy()
    test_df["split"] = "test"
    rows.append(test_df)

splits_df = pd.concat(rows, ignore_index=True)
pd.set_option('display.max_columns', None)
print(splits_df)
print(splits_df.shape)
print(splits_df["split"].value_counts())

splits_df.to_parquet("selected_100_normalised_ph.parquet", index=False)

pd.DataFrame({
    "global_temp_min": [global_temp_min],
    "global_temp_max": [global_temp_max],
    "global_hum_min": [global_hum_min],
    "global_hum_max": [global_hum_max],
}).to_csv("global_weather_scaler.csv", index=False)

house_kwh_scalers.to_csv("local_kwh_scaler.csv", index=False)





