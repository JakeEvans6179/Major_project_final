import subprocess
import sys
from pathlib import Path
import pandas as pd

"""
Master script for extending clustered LSTM64x32 FL training beyond the original 200 rounds.
Starts from chunk 040 and continues to a later final chunk.
"""

ASSIGNMENT_FILE = Path("kmeans_assignments_rowz_k2.csv")

CHUNK_ROUNDS = 5
FRACTION_FIT = 0.3
FRACTION_EVALUATE = 0.0

START_CHUNK = 40     # original training ended here
FINAL_CHUNK = 100    # continue until this chunk number

assignments_df = pd.read_csv(ASSIGNMENT_FILE)
cluster_ids = sorted(assignments_df["cluster"].unique())

for target_cluster in cluster_ids:
    print("\n==============================")
    print(f"Starting extended-rounds training for cluster {target_cluster}")
    print("==============================")

    CHECKPOINT_DIR = Path(f"chunk_checkpoints_cluster_{target_cluster}")
    LOG_ROOT = Path(f"chunk_logs_cluster_{target_cluster}_extended")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_ROOT.mkdir(exist_ok=True)

    current_model = CHECKPOINT_DIR / f"chunk_{START_CHUNK:03d}_LSTM64x32_cluster_{target_cluster}.keras"
    if not current_model.exists():
        raise FileNotFoundError(f"Starting checkpoint not found: {current_model}")

    start_round = START_CHUNK * CHUNK_ROUNDS   # 40 * 5 = 200

    for chunk_idx in range(START_CHUNK + 1, FINAL_CHUNK + 1):
        out_model = CHECKPOINT_DIR / f"chunk_{chunk_idx:03d}_LSTM64x32_cluster_{target_cluster}.keras"
        out_dir = LOG_ROOT / f"chunk_{chunk_idx:03d}"

        cmd = [
            sys.executable,
            "lstm64x32_fl_chunk_run.py",
            "--chunk-rounds", str(CHUNK_ROUNDS),
            "--fraction-fit", str(FRACTION_FIT),
            "--fraction-evaluate", str(FRACTION_EVALUATE),
            "--chunk-index", str(chunk_idx),
            "--start-round", str(start_round),
            "--assignment-file", str(ASSIGNMENT_FILE),
            "--target-cluster", str(target_cluster),
            "--out-model", str(out_model),
            "--out-dir", str(out_dir),
            "--in-model", str(current_model),
        ]

        print(f"\n=== Running cluster {target_cluster}, chunk {chunk_idx}/{FINAL_CHUNK} ===")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

        current_model = out_model
        start_round += CHUNK_ROUNDS

    print(f"\nAll extended chunks completed for cluster {target_cluster}.")
    print(f"Final model: {current_model}")