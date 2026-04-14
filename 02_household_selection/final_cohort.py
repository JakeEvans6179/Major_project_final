import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

'''
Build the final cohort of 100 locked households.
'''

# --- 1. PATHS ---
raw_data_path = Path("selected_households_raw.parquet")
original_sample_path = Path("stage_1_100households.parquet")

output_ids_path = Path("final_locked_100_ids.csv")
output_parquet_path = Path("final_locked_100.parquet")

plots_dir = Path("final_qa_plots")
plots_dir.mkdir(exist_ok=True)

# --- 2. LOAD RAW DATA ---
df = pd.read_parquet(raw_data_path)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["LCLid"] = df["LCLid"].astype(str)

# --- 3. LOAD ORIGINAL 100 IDS ---
original_house_ids = (
    pd.read_parquet(original_sample_path)["LCLid"]
    .astype(str)
    .drop_duplicates()
    .tolist()
)

# --- 4. FINAL ORIGINAL REJECT LIST (13) ---
original_reject_ids = [
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
]

# --- 5. ACCEPTED REPLACEMENTS ---
accepted_stage2_ids = [
    "MAC000019",
    "MAC000055",
    "MAC000093",
    "MAC000104",
    "MAC000111",
    "MAC000125",
    "MAC000244",
    "MAC004499",
    "MAC004848",
    "MAC004853",
    "MAC004862",
]

accepted_stage3_ids = [
    "MAC000060",
    "MAC000126",
]

accepted_replacement_ids = accepted_stage2_ids + accepted_stage3_ids

# --- 6. BUILD FINAL 100 IDS ---
kept_original_ids = [hid for hid in original_house_ids if hid not in original_reject_ids]
final_locked_ids = sorted(set(kept_original_ids + accepted_replacement_ids))

# --- 7. SANITY CHECKS ---
print(f"Original sampled IDs: {len(set(original_house_ids))}")
print(f"Kept original IDs: {len(kept_original_ids)}")
print(f"Accepted replacements: {len(accepted_replacement_ids)}")
print(f"Final locked IDs: {len(final_locked_ids)}")

assert len(set(original_house_ids)) == 100, "Original sample is not 100 unique IDs"
assert len(kept_original_ids) == 87, f"Expected 87 kept originals, got {len(kept_original_ids)}"
assert len(accepted_replacement_ids) == 13, f"Expected 13 replacements, got {len(accepted_replacement_ids)}"
assert len(final_locked_ids) == 100, f"Expected 100 final IDs, got {len(final_locked_ids)}"

# --- 8. BUILD FINAL PARQUET FROM FULL RAW DATA ---
final_locked_df = df[df["LCLid"].isin(final_locked_ids)].copy()

# Optional extra check
final_unique_ids = sorted(final_locked_df["LCLid"].drop_duplicates().tolist())
assert len(final_unique_ids) == 100, f"Expected 100 unique IDs in final parquet, got {len(final_unique_ids)}"

# --- 9. SAVE ---
pd.DataFrame({"LCLid": final_locked_ids}).to_csv(output_ids_path, index=False)
final_locked_df.to_parquet(output_parquet_path, index=False)

print(f"Saved final ID list to: {output_ids_path}")
print(f"Saved final locked parquet to: {output_parquet_path}")

# --- 10. PLOT FINAL QA ---
for i, house_id in enumerate(final_locked_ids, start=1):
    house_df = final_locked_df[final_locked_df["LCLid"] == house_id].sort_values("DateTime")

    plt.figure(figsize=(16, 4))
    plt.plot(house_df["DateTime"], house_df["kwh"], lw=0.5, color="royalblue")
    plt.title(f"QA Inspection Profile: {house_id}")
    plt.xlabel("Date")
    plt.ylabel("Raw kWh")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(plots_dir / f"{house_id}_qa.png", dpi=150)
    plt.close()

    if i % 10 == 0 or i == len(final_locked_ids):
        print(f"Plotted {i}/{len(final_locked_ids)} houses...")

print("Done")