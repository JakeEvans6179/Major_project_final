from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Sample 10 unseen houses not used in the final locked 100-house cohort.
"""

# --- 1. PATHS ---
raw_data_path = Path("../01_data_preparation/eligible_households_raw.parquet")
final_locked_parquet_path = Path("final_locked_100.parquet")

output_ids_path = Path("unseen_house_sample_ids.csv")
output_parquet_path = Path("unseen_house_sample.parquet")

plots_dir = Path("unseen_house_qa_plots")
plots_dir.mkdir(exist_ok=True)

SEED = 6769
N_SAMPLE = 10

# --- 2. LOAD DATA ---
df = pd.read_parquet(raw_data_path)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["LCLid"] = df["LCLid"].astype(str)

final_locked_df = pd.read_parquet(final_locked_parquet_path)
final_locked_ids = (
    final_locked_df["LCLid"]
    .astype(str)
    .drop_duplicates()
    .tolist()
)

# --- 3. MANUAL BAD / REJECT IDS TO ALSO EXCLUDE ---
bad_ids = [
    "MAC000020",
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
    "MAC000057",
    "MAC004575",
]

exclude_ids = sorted(set(final_locked_ids) | set(bad_ids))

# --- 4. FIND ELIGIBLE UNSEEN HOUSES ---

print(f"Final locked cohort IDs: {len(final_locked_ids)}")
print(f"Bad/rejected IDs excluded: {len(set(bad_ids))}")
print(f"Total excluded IDs: {len(exclude_ids)}")


# --- HARD FILTER METRICS ---
print("Calculating hard-filter metrics...")

house_stats = df.groupby("LCLid").agg(
    total_readings=("kwh", "count"),
    zeros_count=("kwh", lambda x: (x == 0).sum()),
    max_kwh=("kwh", "max"),
    std_kwh=("kwh", "std"),
    unique_values=("kwh", "nunique")
).reset_index()

house_stats["zeros_ratio"] = house_stats["zeros_count"] / house_stats["total_readings"]

exclude_ids = sorted(set(final_locked_ids) | set(bad_ids))

valid_houses_df = house_stats[
    (house_stats["zeros_ratio"] <= 0.05) &
    (house_stats["std_kwh"] >= 0.05) &
    (house_stats["max_kwh"] <= 10.0) &
    (house_stats["unique_values"] > 500) &
    (~house_stats["LCLid"].isin(exclude_ids))
].copy()

print(f"Total houses analysed: {len(house_stats)}")
print(f"Excluded final cohort + bad IDs: {len(exclude_ids)}")
print(f"Houses surviving hard filters and exclusions: {len(valid_houses_df)}")

if len(valid_houses_df) < N_SAMPLE:
    raise ValueError(f"Not enough eligible unseen houses to sample {N_SAMPLE}.")

# --- 5. RANDOM SAMPLE ---
rng = np.random.default_rng(SEED)
sampled_ids = sorted(
    rng.choice(valid_houses_df["LCLid"].to_numpy(), size=N_SAMPLE, replace=False)
)

print(f"\nSampled unseen houses ({N_SAMPLE}, seed={SEED}):")
print(sampled_ids)

# --- 6. SAVE IDS + PARQUET ---
sampled_df = df[df["LCLid"].isin(sampled_ids)].copy()

pd.DataFrame({"LCLid": sampled_ids}).to_csv(output_ids_path, index=False)
sampled_df.to_parquet(output_parquet_path, index=False)

print(f"Saved sampled IDs to: {output_ids_path}")
print(f"Saved sampled parquet to: {output_parquet_path}")

# --- 7. QA PLOTS ---
for i, house_id in enumerate(sampled_ids, start=1):
    house_df = sampled_df[sampled_df["LCLid"] == house_id].sort_values("DateTime")

    plt.figure(figsize=(16, 4))
    plt.plot(house_df["DateTime"], house_df["kwh"], lw=0.5, color="royalblue")
    plt.title(f"QA Inspection Profile: {house_id}")
    plt.xlabel("Date")
    plt.ylabel("Raw kWh")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{house_id}_qa.png", dpi=150)
    plt.close()

    if i % 5 == 0 or i == len(sampled_ids):
        print(f"Plotted {i}/{len(sampled_ids)} houses...")

print("Done.")