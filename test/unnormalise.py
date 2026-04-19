from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


import random
import Helper_functions

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

'''
unnormalise house data
'''



data_path = Path("final_locked_100_normalised.parquet")

max_min_path = Path("global_weather_scaler.csv")

local_kwh_scaling = Path("local_kwh_scaler.csv")


df, local_kwh_scaler_df, global_temp_min, global_temp_max, global_hum_min, global_hum_max = Helper_functions.load_data(data_path, max_min_path, local_kwh_scaling)   #load global weather scalers and local kwh scalers

house_ids = sorted(df["LCLid"].unique())


unscaled_houses = []

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    # Extract the local max/min
    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)

    # Isolate the house
    house_df = df[df["LCLid"] == house_id].copy()

    # Un-normalise the kwh column back to real physical values
    house_df["kwh"] = Helper_functions.unscale(house_df["kwh"], kwh_min, kwh_max)

    # Filter down to strictly the columns you asked for
    house_df_minimal = house_df[["DateTime", "LCLid", "kwh"]]

    # Append to our master list
    unscaled_houses.append(house_df_minimal)

# Recombine all 100 houses into a single DataFrame
final_raw_df = pd.concat(unscaled_houses, ignore_index=True)

print("Unscaling complete. Final shape:", final_raw_df.shape)
print(final_raw_df.head())


final_raw_df.to_parquet("final_locked_100_RAW.parquet", index=False)


# --- SANITY CHECK ---
print("\n--- Running Sanity Check ---")
errors = []

for house_id in house_ids:
    #Grab the original scaled array and our new raw array
    original_scaled = df[df["LCLid"] == house_id]["kwh"].values
    new_raw = final_raw_df[final_raw_df["LCLid"] == house_id]["kwh"].values
    
    #Grab the local scalers
    kwh_min, kwh_max = Helper_functions.extract_kwh(local_kwh_scaler_df, house_id)
    
    #Manually re-scale the raw data back to [0, 1]
    
    rescaled = (new_raw - kwh_min) / (kwh_max - kwh_min)
    
    # 4. Find the absolute largest difference between the original and our re-scaled version
    max_diff = np.max(np.abs(original_scaled - rescaled))
    errors.append(max_diff)

global_max_error = max(errors)
print(f"Maximum variance across all 800,000+ rows: {global_max_error}")

# Note: We check against 1e-10 instead of 0.0 because Python floating-point 
# math occasionally leaves microscopic artifacts like 0.0000000000000002
if global_max_error < 1e-10:
    print("checks passed")
else:
    print("check failed, not the same")

    

    
    






