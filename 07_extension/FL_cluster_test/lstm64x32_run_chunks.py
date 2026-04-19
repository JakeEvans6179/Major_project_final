import subprocess
import sys
from pathlib import Path

'''

Master script for running LSTM 64x32 FL training in chunks on specified cluster. 
Change assignment file and target cluster variables depending on which cluster is being run on

'''
#Cluster settings
ASSIGNMENT_FILE = Path("kmeans_assignments_rowz_k2.csv") #household ids assigned to clusters (change depending number of clusters)
target_cluster = 1  #specifiy target cluster for run (change depending on cluster being run on)

# ===== chunk settings =====
TOTAL_CHUNKS = 40
CHUNK_ROUNDS = 5
FRACTION_FIT = 0.3
FRACTION_EVALUATE = 0.0   #no eval, eval done in separate script

CHECKPOINT_DIR = Path(f"chunk_checkpoints_cluster_{target_cluster}")
LOG_ROOT = Path(f"chunk_logs_cluster_{target_cluster}")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_ROOT.mkdir(exist_ok=True)



current_model = ""
start_round = 0

for chunk_idx in range(1, TOTAL_CHUNKS + 1):
    out_model = CHECKPOINT_DIR / f"chunk_{chunk_idx:03d}_LSTM64x32_cluster_{target_cluster}.keras"
    out_dir = LOG_ROOT / f"chunk_{chunk_idx:03d}"
    assignment_file = ASSIGNMENT_FILE

    cmd = [
        sys.executable,
        "lstm64x32_fl_chunk_run.py",
        "--chunk-rounds", str(CHUNK_ROUNDS),
        "--fraction-fit", str(FRACTION_FIT),
        "--fraction-evaluate", str(FRACTION_EVALUATE),
        "--chunk-index", str(chunk_idx),
        "--start-round", str(start_round),
        "--assignment-file", str(assignment_file),
        "--target-cluster", str(target_cluster),
        "--out-model", str(out_model),
        "--out-dir", str(out_dir),
    ]

    if current_model:
        cmd += ["--in-model", current_model]

    print(f"\n=== Running chunk {chunk_idx}/{TOTAL_CHUNKS} ===")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    current_model = str(out_model)
    start_round += CHUNK_ROUNDS

print("\nAll chunks completed.")
print(f"Final model: {current_model}")