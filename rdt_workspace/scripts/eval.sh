#!/bin/bash
#SBATCH --job-name=RDT_EVAL
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/e12434694/rdt_workspace/logs/eval_%j.log

# 1. Environment Setup
source /home/e12434694/miniconda3/etc/profile.d/conda.sh
conda activate rdt_env

# 2. Silence the Noise & Fix Headless Rendering
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error
export SAPIEN_NO_GUI=1
export PYOPENGL_PLATFORM=egl

# 3. Offline Weights Setup
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/home/e12434694/RoboticsDiffusionTransformer"

# 4. Vulkan & Rendering Fix
if [ -f "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json" ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
elif [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR

# 5. Library Links
export LD_LIBRARY_PATH=/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH
# Preloading libcuda often helps with headless gym environments
export LD_PRELOAD=/lib64/libcuda.so.1

# ... (Previous parts remain the same) ...

# 6. Execution
cd /home/e12434694/RoboticsDiffusionTransformer/
export PYTHONPATH=$PYTHONPATH:$(pwd)



# Fix the /tmp error by using your home dir instead
export XDG_RUNTIME_DIR=/home/e12434694/tmp_runtime
mkdir -p $XDG_RUNTIME_DIR

echo "--- Starting RDT Evaluation on $(hostname) ---"
# We define an environment variable for the config path
export RDT_CONFIG_PATH="/home/e12434694/rdt_workspace/configs/base.yaml"

python -u /home/e12434694/rdt_workspace/scripts/evaluation.py \
    --pretrained_path /home/e12434694/rdt_workspace/checkpoints/checkpoint-5000/ema/model.safetensors\
    -e "PickCube-v1" \
    --sim-backend "gpu" \
    -n 25