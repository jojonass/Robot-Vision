#!/bin/bash
#SBATCH --job-name=RDT_FINETUNE
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/e12434694/rdt_workspace/logs/train_%j.log

# 1. Paths
REPO_DIR="/home/e12434694/RoboticsDiffusionTransformer"
WORKSPACE="/home/e12434694/rdt_workspace"
CONDA_ENV_BIN="/home/e12434694/miniconda3/envs/rdt_env/bin/python3"
CONDA_PKGS="/home/e12434694/miniconda3/envs/rdt_env/lib/python3.10/site-packages"

# 2. Environment Setup
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
# COMBINE paths into one export so they don't overwrite
export APPTAINERENV_PYTHONPATH="$CONDA_PKGS:$WORKSPACE/scripts:$REPO_DIR:/zed_libs"

cd "$REPO_DIR"

echo "[START] Launching RDT Fine-tuning at 25Hz..."

apptainer exec --nv --no-home \
    -B "/home/e12434694/zed_container_libs:/zed_libs,/home/e12434694:/home/e12434694" \
    /home/e12434694/zed_sdk.sif \
    $CONDA_ENV_BIN -m accelerate.commands.launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision="bf16" \
    --dynamo_backend="no" \
    ./main.py \
    --deepspeed="./configs/zero2.json" \
    --config_path="/home/e12434694/rdt_workspace/scripts/configs/base.yaml" \
    --pretrained_model_name_or_path="./rdt_1b" \
    --pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl" \
    --pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384" \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --sample_batch_size=4 \
    --dataloader_num_workers=8 \
    --learning_rate=5e-5 \
    --output_dir="/home/e12434694/rdt_workspace/checkpoints" \
    --max_train_steps=5000 \
    --checkpointing_period=5000