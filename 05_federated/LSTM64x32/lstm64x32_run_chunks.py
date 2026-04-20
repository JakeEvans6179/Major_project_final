import subprocess
import sys
from pathlib import Path

CHUNK_ROUNDS = 5
FRACTION_FIT = 0.3
FRACTION_EVALUATE = 0.0

START_CHUNK = 40
FINAL_CHUNK = 100   

CHECKPOINT_DIR = Path("chunk_checkpoints")
LOG_ROOT = Path("chunk_logs_extended")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_ROOT.mkdir(exist_ok=True)

current_model = CHECKPOINT_DIR / f"global_chunk_{START_CHUNK:03d}_LSTM64x32.keras"
start_round = START_CHUNK * CHUNK_ROUNDS   # 40 * 5 = 200

if not current_model.exists():
    raise FileNotFoundError(f"Starting checkpoint not found: {current_model}")

for chunk_idx in range(START_CHUNK + 1, FINAL_CHUNK + 1):
    out_model = CHECKPOINT_DIR / f"global_chunk_{chunk_idx:03d}_LSTM64x32.keras"
    out_dir = LOG_ROOT / f"chunk_{chunk_idx:03d}"

    cmd = [
        sys.executable,
        "lstm64x32_fl_chunk_run.py",
        "--chunk-rounds", str(CHUNK_ROUNDS),
        "--fraction-fit", str(FRACTION_FIT),
        "--fraction-evaluate", str(FRACTION_EVALUATE),
        "--chunk-index", str(chunk_idx),
        "--start-round", str(start_round),
        "--out-model", str(out_model),
        "--out-dir", str(out_dir),
        "--in-model", str(current_model),
    ]

    print(f"\n=== Running chunk {chunk_idx}/{FINAL_CHUNK} ===")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    current_model = out_model
    start_round += CHUNK_ROUNDS

print("\nAll chunks completed.")
print(f"Final model: {current_model}")