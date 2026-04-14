import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

'''
Sample new replacement candidates for stage 3.
'''
# --- 1. CONFIGURATION ---
data_path = Path("../01_data_preparation/eligible_households_raw.parquet")              # full raw candidate pool
original_sample_path = Path("stage_1_100households.parquet")     # raw data for original sampled 100
stage2_ids_path = Path("stage2_replacement_candidates.csv")      # 22 stage 2 candidate IDs
plots_dir = Path("stage3_qa_plots")
plots_dir.mkdir(exist_ok=True)
SEED = 6769

# --- 2. LOAD DATA ---
print("Loading raw dataset...")
df = pd.read_parquet(data_path)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["LCLid"] = df["LCLid"].astype(str)

# Original sampled 100 IDs
original_house_ids = (
    pd.read_parquet(original_sample_path)["LCLid"]
    .astype(str)
    .drop_duplicates()
    .tolist()
)

# Stage 2 replacement candidate IDs (the 22 candidates you reviewed)
stage2_ids = (
    pd.read_csv(stage2_ids_path)["LCLid"]
    .astype(str)
    .tolist()
)

# All rejected IDs so far:
# - original 22 rejects
# - rejected stage 2 replacements
BAD_IDS = [
    "MAC000020",    #stage 1 rejects
    "MAC000023",
    "MAC000072",  
    "MAC000222",
    "MAC000229",   
    "MAC004476",
    "MAC004487",
    "MAC004488",
    "MAC004518",
    "MAC004552",
    "MAC004579",
    "MAC004866",
    "MAC004863",

    "MAC000057",   #stage 2 rejects
    "MAC004575",
]

#Approved Stage 2 replacements = stage 2 candidates that are NOT in BAD_IDS

approved_s2_ids = []

for house_id in stage2_ids:
    if house_id not in BAD_IDS:
        print(f"Approved Stage 2 replacement: {house_id}")
        approved_s2_ids.append(house_id)

num_resamples = 13 - len(approved_s2_ids)

print(f"Original sampled households: {len(original_house_ids)}")
print(f"Approved Stage 2 replacements kept: {len(approved_s2_ids)}")
print(f"New replacements needed now: {num_resamples}")

# --- 3. CALCULATE HARD-FILTER METRICS ---
print("Calculating hard-filter metrics...")
house_stats = df.groupby("LCLid").agg(
    total_readings=("kwh", "count"),
    zeros_count=("kwh", lambda x: (x == 0).sum()),
    max_kwh=("kwh", "max"),
    std_kwh=("kwh", "std"),
    unique_values=("kwh", "nunique")
).reset_index()

house_stats["zeros_ratio"] = house_stats["zeros_count"] / house_stats["total_readings"]

# Apply hard filters + exclusion rules
valid_houses_df = house_stats[
    (house_stats["zeros_ratio"] <= 0.05) &
    (house_stats["std_kwh"] >= 0.05) &
    (house_stats["max_kwh"] <= 10.0) &
    (house_stats["unique_values"] > 500) &
    (~house_stats["LCLid"].isin(BAD_IDS)) &
    (~house_stats["LCLid"].isin(original_house_ids)) &
    (~house_stats["LCLid"].isin(approved_s2_ids))
].copy()

print(f"\nTotal houses analysed: {len(house_stats)}")
print(f"Houses surviving hard filters and exclusions: {len(valid_houses_df)}")

if len(valid_houses_df) < num_resamples:
    raise ValueError(
        f"Not enough valid houses left to sample {num_resamples} replacements. "
        f"Only {len(valid_houses_df)} available."
    )

valid_houses_df = valid_houses_df.sort_values("LCLid").reset_index(drop=True)
# --- 4. SAMPLE NEW REPLACEMENT CANDIDATES ---
rng = np.random.default_rng(seed=SEED)

stage3_resampled_ids = sorted(
    rng.choice(valid_houses_df["LCLid"].to_numpy(), size=num_resamples, replace=False)
)

print(f"\nSampled {len(stage3_resampled_ids)} new replacement candidates (Seed: {SEED}).")

pd.DataFrame({"LCLid": stage3_resampled_ids}).to_csv(
    "stage3_replacement_candidates.csv", index=False
)
print("Saved replacement IDs to 'stage3_replacement_candidates.csv'.")

# Optional: save approved Stage 2 replacements for bookkeeping
pd.DataFrame({"LCLid": approved_s2_ids}).to_csv(
    "approved_stage2_replacement_ids.csv", index=False
)

# --- 5. PLOT FOR MANUAL CHECK ---
print("\nGenerating QA plots for visual inspection...")
replacement_raw_df = df[df["LCLid"].isin(stage3_resampled_ids)].copy()

for i, house_id in enumerate(stage3_resampled_ids, start=1):
    house_df = replacement_raw_df[replacement_raw_df["LCLid"] == house_id].sort_values("DateTime")

    plt.figure(figsize=(16, 4))
    plt.plot(house_df["DateTime"], house_df["kwh"], lw=0.5, color="royalblue")
    plt.title(f"QA Inspection Profile: {house_id}")
    plt.xlabel("Date")
    plt.ylabel("Raw kWh")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(plots_dir / f"{house_id}_qa.png", dpi=150)
    plt.close()

    if i % 10 == 0 or i == len(stage3_resampled_ids):
        print(f"Plotted {i}/{len(stage3_resampled_ids)} houses...")

print("Done.")

pd.DataFrame({"LCLid": sorted(BAD_IDS)}).to_csv("all_rejected_ids_stage3.csv", index=False)

replacement_raw_df.to_parquet("stage3_replacement_candidates.parquet", index=False)
print("Saved replacement raw data to 'stage3_replacement_candidates.parquet'.")