#!/bin/bash
#SBATCH --job-name=RDT_PREP
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/e12434694/rdt_workspace/logs/prep_%j.log

# Updated to miniconda3 based on your logs
source /home/e12434694/miniconda3/etc/profile.d/conda.sh
conda activate rdt_env

python3 /home/e12434694/rdt_workspace/scripts/prep_all.py