
````markdown
# RDT-DE

RDT-DE provides the environment setup and cluster execution pipeline for running evaluation with the ZED SDK and DROID SVO trajectories on a SLURM-based GPU cluster.

The project uses:
- A Conda environment (`rdt_env`) for Python dependencies
- An Apptainer container for CUDA and ZED SDK support
- A cluster patch strategy to bridge hardware and external Python libraries

---

# Prerequisite: RDT-1B Reproduction

This repository assumes that **RDT-1B** has already been correctly set up.

Before continuing, you must:

1. Visit the official **RDT-1B GitHub repository**
2. Follow their instructions to:
   - Download the pretrained model weights
   - Download the required encoders
3. Place the downloaded files exactly as specified in the original RDT-1B directory structure

Important:
- Keep checkpoint files in their expected directories (e.g., `rdt_1b/`)
- Do not rename weight files
- Ensure encoder files are placed in the correct subdirectories

This repository relies on the exact folder structure expected by RDT-1B.  
If the weights or encoders are misplaced, evaluation will fail.

Once RDT-1B is correctly set up, continue with the environment configuration below.

---

# 1. Environment Setup

To ensure reproducibility, recreate the Conda environment using the provided `environment.yml`.

### Create the Environment

```bash
conda env create -f environment.yml
````

### Activate the Environment

```bash
conda activate rdt_env
```

(Optional) Verify installation:

```bash
python --version
conda list
```

The Python environment is now ready.

---

## Running the RDT-1B Reproduction Benchmark

To reproduce the baseline benchmark used for testing RDT-1B, use a SLURM script similar to the following:

```bash
#!/bin/bash
#SBATCH --job-name=rdt_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=rdt_eval_%j.out

# Activate environment
source /home/miniconda3/etc/profile.d/conda.sh
conda activate rdt_env

# Silence warnings
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export SAPIEN_NO_GUI=1

# Offline mode
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/home/RoboticsDiffusionTransformer"

# Vulkan configuration
if [ -f "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json" ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
elif [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR

# Library configuration
export LD_LIBRARY_PATH=/lib64:/usr/lib64/nvidia:$LD_LIBRARY_PATH
export LD_PRELOAD=/lib64/libcuda.so.1

# Run evaluation
cd /home/RoboticsDiffusionTransformer/
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u eval_sim/eval_rdt_maniskill.py \
    --pretrained_path /home/RoboticsDiffusionTransformer/rdt_1b/mp_rank_00_model_states.pt \
    -e "PickCube-v1" \
    -o "rgb" \
    --sim-backend "gpu" \
    -n 25
```

---

# 2. Data Extraction Pipeline (Cluster Setup)

The data extraction and evaluation pipeline runs inside an Apptainer container due to cluster restrictions.

Because the container image is read-only, we use a **Cluster Patch strategy**:

* The base container provides Ubuntu, CUDA, and the ZED C++ SDK.
* External Python dependencies are mounted at runtime.
* USB drivers are manually bridged.

---

## Step 1 — Pull the ZED Apptainer Image

```bash
apptainer pull ~/zed_sdk.sif docker://stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04
```

The `gl-devel` version is required for EGL-based off-screen rendering.

---

## Step 2 — Prepare the Python Patch (Local Machine)

Create a portable Python package directory locally and upload it to the cluster.

```bash
mkdir -p zed_container_libs
```

Install required dependencies:

```bash
pip install --target=./zed_container_libs \
    numpy==1.24.4 \
    opencv-python \
    h5py \
    pyrallis \
    tqdm \
    google-cloud-storage \
    google-auth \
    protobuf \
    rsa \
    pyasn1
```

Install the ZED Python wrapper (Python 3.8):

```bash
pip install --target=./zed_container_libs \
    https://download.stereolabs.com/zedsdk/4.0/whl/linux_x86_64/pyzed-4.0-cp38-cp38-linux_x86_64.whl
```

Upload the full `zed_container_libs` directory to your cluster home.

---

## Step 3 — Hardware Bridge & Authentication

Extract the USB bridge (run once on cluster):

```bash
apptainer exec ~/zed_sdk.sif \
    cp /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 ~/libusb-1.0.so.0
```

For DROID `.svo` downloads, place your service account key (`credentials.json`) in the project root and export:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
```

---

## Step 4 — Running via SLURM

Example execution script:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1

PATCH_LIBS="/home/$USER/zed_container_libs"
USB_BRIDGE="/home/$USER/libusb-1.0.so.0"
export GOOGLE_APPLICATION_CREDENTIALS="./credentials.json"

apptainer exec --nv \
    --bind "$USB_BRIDGE":/usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 \
    --bind "$PATCH_LIBS":/usr/local/python_patch \
    --env PYTHONPATH="/usr/local/python_patch:$PYTHONPATH" \
    ~/zed_sdk.sif python3 -u scripts/evaluation.py \
        --pretrained_path ./checkpoints/checkpoint-5000/ema/model.safetensors \
        --sim-backend "gpu"
```

---

# 3. rdt_workspace Directory

The main project code is located in the `rdt_workspace` directory.

### Python Scripts

* `evaluation.py` — Runs inference and computes success rates for RDT inside ManiSkill.
* `hdf5_maniskill_dataset.py` — Defines data loading for ManiSkill HDF5 trajectories.
* `prep_all.py` — Executes the full data preparation pipeline.
* `svo_online.py` — Extracts depth data from ZED `.svo` files.
* `unique_inst.py` — Processes and filters natural language instructions.

### SLURM / Shell Scripts

* `finetune_maniskill.sh` — Launches RDT-DE fine-tuning on the cluster.
* `eval.sh` — Submits evaluation jobs.
* `submit_prep.sh` — Submits data preprocessing jobs.
* `submit_droid.sh` — Processes raw DROID dataset files.
* `submit_instructions.sh` — Submits instruction filtering jobs.

---

# Setup Summary

1. Reproduce RDT-1B (download weights and encoders)
2. Create the Conda environment (`rdt_env`)
3. Pull the ZED Apptainer image
4. Create and upload the Python patch
5. Extract the USB bridge
6. Provide Google Cloud credentials
7. Run the SLURM job

```

---


