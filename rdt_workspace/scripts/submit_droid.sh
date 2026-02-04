#!/bin/bash
#SBATCH --job-name=DROID_PARALLEL
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=1-8
#SBATCH --output=/home/e12434694/rdt_workspace/logs/prep_%j.log

mkdir -p logs

# 1. ZED Settings path
ZED_LOCAL_PATH="/home/e12434694/.zed_settings"
mkdir -p "$ZED_LOCAL_PATH"

# 2. Environment Setup
unset PYTHONPATH
# We bind the libs folder to a generic internal path /zed_libs
export APPTAINER_BIND="/home/e12434694/zed_container_libs:/zed_libs,/home/e12434694:/home/e12434694"
export APPTAINERENV_PYTHONPATH="/zed_libs"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_ZED_SETTINGS_PATH="$ZED_LOCAL_PATH"

# 3. Calculation
ACTUAL_ID=$((${SLURM_ARRAY_TASK_ID} - 1))

echo "[START] Worker ${SLURM_ARRAY_TASK_ID} (ID: ${ACTUAL_ID}) starting..."

# 4. Execution
# Note: Bindings are handled by the export above for cleaner syntax
apptainer exec --nv --no-home \
    /home/e12434694/zed_sdk.sif \
    python3 /home/e12434694/rdt_workspace/scripts/svo_online.py\
    --worker_id ${ACTUAL_ID} \
    --num_workers 8

echo "[END] Worker ${SLURM_ARRAY_TASK_ID} finished."