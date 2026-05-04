import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

'''
load eligible houses, check against hard filters, sample 100, and plot for manual check
'''

#get raw elgible households data
data_path = Path("../01_data_preparation/eligible_households_raw.parquet")
plots_dir = Path("stage1_qa_plots")
plots_dir.mkdir(exist_ok=True)
SEED = 6769

#load data
print("Loading raw dataset...")
df = pd.read_parquet(data_path)
df["DateTime"] = pd.to_datetime(df["DateTime"])

#calculate metrics for filtering step later 
print("Calculating hardware and physical heuristics...")
house_stats = df.groupby("LCLid").agg(
    total_readings=("kwh", "count"),
    zeros_count=("kwh", lambda x: (x == 0).sum()),
    max_kwh=("kwh", "max"),
    std_kwh=("kwh", "std"),
    unique_values=("kwh", "nunique")
).reset_index()

house_stats["zeros_ratio"] = house_stats["zeros_count"] / house_stats["total_readings"]

#hard filter rules
valid_houses_df = house_stats[
    (house_stats["zeros_ratio"] <= 0.05) &      #no long grid disconnection
    (house_stats["std_kwh"] >= 0.05) &          #needs plausible variability
    (house_stats["max_kwh"] <= 10.0) &          #implausible high readings, may show faulty sensor/ data issues
    (house_stats["unique_values"] > 500)        #no stuck readings, needs some varaibility in data values
].copy()

print(f"\nTotal Houses Analyzed: {len(house_stats)}")
print(f"Houses Surviving Hard Filters: {len(valid_houses_df)}")

valid_houses_df = valid_houses_df.sort_values("LCLid").reset_index(drop=True)
#sample 100 random households
rng = np.random.default_rng(seed=SEED)

if len(valid_houses_df) < 100:
    raise ValueError("less than 100 sampled households")

#sort by ascending order of houseid
initial_100_ids = sorted(rng.choice(valid_houses_df["LCLid"].to_numpy(), size=100, replace=False))
print(f"\nSampled initial 100 households (Seed: {SEED}).")


pd.DataFrame({"LCLid": initial_100_ids}).to_csv("stage1_initial_100.csv", index=False)  #save ids to csv
print("Saved initial list to 'stage1_initial_100.csv'.")

#plot for manual check
print(f"\nplotting for visual inspection")
final_raw_df = df[df["LCLid"].isin(initial_100_ids)].copy()

for i, house_id in enumerate(initial_100_ids, start=1):
    house_df = final_raw_df[final_raw_df["LCLid"] == house_id].sort_values("DateTime")
    
    plt.figure(figsize=(16, 4))
    plt.plot(house_df["DateTime"], house_df["kwh"], lw=0.5, color="royalblue")
    plt.title(f"QA Inspection Profile: {house_id}")
    plt.xlabel("Date")
    plt.ylabel("Raw kWh")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plots_dir / f"{house_id}_qa.png", dpi=150)
    plt.close()
    
    if i % 10 == 0 or i == len(initial_100_ids):
        print(f"Plotted {i}/{len(initial_100_ids)} houses")

print("Done")

final_raw_df.to_parquet("stage_1_100households.parquet")