import subprocess
import sys
from pathlib import Path

'''

Master script for running S2S FL training in chunks on specified cluster. 
Change assignment file and target cluster variables depending on which cluster is being run on

'''
#Cluster settings
import subprocess
import sys
from pathlib import Path
import pandas as pd

ASSIGNMENT_FILE = Path("kmeans_assignments_rowu_k4.csv")

TOTAL_CHUNKS = 40
CHUNK_ROUNDS = 5
FRACTION_FIT = 0.3
FRACTION_EVALUATE = 0.0

assignments_df = pd.read_csv(ASSIGNMENT_FILE)
cluster_ids = sorted(assignments_df["cluster"].unique())


for target_cluster in cluster_ids:
    print(f"\n==============================")
    print(f"Starting training for cluster {target_cluster}")
    print(f"==============================")

    CHECKPOINT_DIR = Path(f"chunk_checkpoints_cluster_{target_cluster}")
    LOG_ROOT = Path(f"chunk_logs_cluster_{target_cluster}")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_ROOT.mkdir(exist_ok=True)

    current_model = ""
    start_round = 0

    for chunk_idx in range(1, TOTAL_CHUNKS + 1):
        out_model = CHECKPOINT_DIR / f"chunk_{chunk_idx:03d}_S2S_cluster_{target_cluster}.keras"
        out_dir = LOG_ROOT / f"chunk_{chunk_idx:03d}"

        cmd = [
            sys.executable,
            "s2s_fl_chunk_run.py",
            "--chunk-rounds", str(CHUNK_ROUNDS),
            "--fraction-fit", str(FRACTION_FIT),
            "--fraction-evaluate", str(FRACTION_EVALUATE),
            "--chunk-index", str(chunk_idx),
            "--start-round", str(start_round),
            "--assignment-file", str(ASSIGNMENT_FILE),
            "--target-cluster", str(target_cluster),
            "--out-model", str(out_model),
            "--out-dir", str(out_dir),
        ]

        if current_model:
            cmd += ["--in-model", current_model]

        print(f"\n=== Running cluster {target_cluster}, chunk {chunk_idx}/{TOTAL_CHUNKS} ===")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

        current_model = str(out_model)
        start_round += CHUNK_ROUNDS

    print(f"\nAll chunks completed for cluster {target_cluster}.")
    print(f"Final model: {current_model}")