#!/bin/bash
#SBATCH --job-name=DROID_SAMPLE
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8      # Increased for better I/O handling
#SBATCH --mem=48G              # Increased to prevent OOM during list shuffling
#SBATCH --time=04:00:00 
#SBATCH --output=logs/sample_audit_%j.log

mkdir -p logs STATS

# Ensure the list exists; if not, generate it on the host first
LIST_FILE="/home/e12434694/STATS/all_metadata.txt"
if [ ! -f "$LIST_FILE" ]; then
    echo "Metadata list not found. Generating..."
    gsutil ls "gs://gresearch/robotics/droid_raw/1.0.1/**/metadata_*.json" > "$LIST_FILE"
fi

# Environment Setup: Pass the host's PATH to the container so it can find gsutil
export APPTAINER_BIND="/home/e12434694:/home/e12434694"
export APPTAINERENV_PATH="$PATH"

echo "[START] Starting Random Audit of 7600 files..."

# Execution
apptainer exec --nv --no-home \
    /home/e12434694/zed_sdk.sif \
    python3 /home/e12434694/test_online_stats.py \
    --list_path "$LIST_FILE" \
    --sample_size 7600

echo "[END] Audit complete."