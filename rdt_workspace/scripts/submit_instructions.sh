#!/bin/bash
#SBATCH --job-name=DROID_AUDIT
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/e12434694/rdt_workspace/logs/prep_%j.log

# 1. Setup
mkdir -p /home/e12434694/rdt_workspace/logs
ZED_LOCAL_PATH="/home/e12434694/.zed_settings"
mkdir -p "$ZED_LOCAL_PATH"

# Paths
WORKSPACE="/home/e12434694/rdt_workspace"

# --- CRITICAL CHANGES ---
# 1. Removed PYTHONNOUSERSITE so it CAN use your .local installs
# 2. Added your script directory to PYTHONPATH
export APPTAINERENV_PYTHONPATH="/zed_libs:$WORKSPACE/scripts"
unset APPTAINERENV_PYTHONNOUSERSITE 

# 3. Add NLTK_DATA path so it saves the thesaurus data to your workspace
export APPTAINERENV_NLTK_DATA="$WORKSPACE/nltk_data"
mkdir -p "$WORKSPACE/nltk_data"

echo "[START] Audit starting..."

# 2. Execution 
# We add -B for your .local folder so the container can see the NLTK you installed
apptainer exec --nv --no-home \
    -B "/home/e12434694/zed_container_libs:/zed_libs" \
    -B "/home/e12434694:/home/e12434694" \
    -B "/home/e12434694/.local:/home/e12434694/.local" \
    /home/e12434694/zed_sdk.sif \
    python3 "$WORKSPACE/scripts/unique_inst.py" \
    --limit -1 \
    --min_samples 10

echo "[END] Audit finished."