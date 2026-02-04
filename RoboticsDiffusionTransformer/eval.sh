#!/bin/bash
#SBATCH --job-name=RDT_EVAL
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=rdt_eval_%j.out

# 1. Environment Setup
source /home/e12434694/miniconda3/etc/profile.d/conda.sh
conda activate rdt_env

# 2. Silence the Noise (Warnings & Logs)
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export SAPIEN_NO_GUI=1

# 3. Offline Weights Setup
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/home/e12434694/RoboticsDiffusionTransformer"

# 4. Vulkan & Rendering Fix (Enables the robot's "eyes")
if [ -f "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json" ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
elif [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR

# 5. Library Links
export LD_LIBRARY_PATH=/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH
export LD_PRELOAD=/lib64/libcuda.so.1

# 6. Execution
cd /home/e12434694/RoboticsDiffusionTransformer/
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- Starting RDT Evaluation on $(hostname) ---"
python -u eval_sim/eval_rdt_maniskill.py \
    --pretrained_path /home/e12434694/RoboticsDiffusionTransformer/rdt_1b/mp_rank_00_model_states.pt \
    -e "PickCube-v1" \
    -o "rgb" \
    --sim-backend "gpu" \
    -n 100