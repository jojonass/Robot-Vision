

* Project name fixed: **RDT-DE**
* Environment name fixed: **rdt_env**


---

````markdown
# RDT-DE

RDT-DE provides the environment setup and cluster execution pipeline for running evaluation with the ZED SDK and DROID SVO trajectories on a SLURM-based GPU cluster.



The project uses:
- A Conda environment for Python dependencies
- An Apptainer container for CUDA and ZED SDK support
- A cluster patch strategy to bridge hardware and Python libraries

---


This repository assumes reproduction of **RDT-1B**. Please complete the following prerequisite step before continuing.

---

# Prerequisite: RDT-1B Reproduction

Before setting up this repository, you must:

1. Visit the official **RDT-1B GitHub repository**
2. Follow their instructions to:
   - Download the pretrained model weights
   - Download the required encoders
3. Place the downloaded files exactly as specified in the RDT-1B repository structure

It is important that:
- The checkpoint files remain in the expected directory (e.g., `checkpoints/`)
- Encoder files are placed in the correct subdirectories
- File names are not modified

This repository relies on the exact folder structure expected by RDT-1B.  
If the weights or encoders are misplaced, evaluation will fail.

Once RDT-1B is correctly set up, proceed with the environment setup below.



# 1. Environment Setup

To ensure reproducibility, recreate the Conda environment using the provided `environment.yml` file.

### Create the Environment

```bash
conda env create -f environment.yml


# 1. Environment Setup

To ensure reproducibility, recreate the Conda environment using the provided `environment.yml` file.

### Create the Environment

```bash
conda env create -f environment.yml
````

### Activate the Environment

```bash
conda activate rdt_env
```

(Optional) Verify the installation:

```bash
python --version
conda list
```

The Python environment is now ready.

---

To Run reproduction benchmark used to test baseline for RDT-1B run this script as an example 


#!/bin/bash
#SBATCH --job-name
#SBATCH --partition
#SBATCH --gres
#SBATCH --cpus-per-task
#SBATCH --mem
#SBATCH --time
#SBATCH --output=rdt_eval_%j.out

# 1. Environment Setup
source /home/miniconda3/etc/profile.d/conda.sh
conda activate rdt_env

# 2. Silence the Noise (Warnings & Logs)
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export SAPIEN_NO_GUI=1

# 3. Offline Weights Setup
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="/home/RoboticsDiffusionTransformer"

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
cd /home/RoboticsDiffusionTransformer/
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- Starting RDT Evaluation on $(hostname) ---"
python -u eval_sim/eval_rdt_maniskill.py \
    --pretrained_path /home/RoboticsDiffusionTransformer/rdt_1b/mp_rank_00_model_states.pt \
    -e "PickCube-v1" \
    -o "rgb" \
    --sim-backend "gpu" \
    -n 25






# 2. Data Extraction Pipeline (Cluster Setup)

The data extraction and evaluation pipeline runs inside an Apptainer container due to cluster restrictions.

Because the container image is read-only, we use a **Cluster Patch strategy**:

* The base container provides CUDA, Ubuntu, and the ZED C++ SDK.
* External Python dependencies are mounted at runtime.
* USB drivers are bridged manually.

---

## Step 1 — Pull the ZED Apptainer Image

On the cluster, pull the required container image:

```bash
apptainer pull ~/zed_sdk.sif docker://stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04
```

The `gl-devel` version is required for EGL-based off-screen rendering.

---

## Step 2 — Prepare the Python Patch (Local Machine)

Since the container is read-only, create a portable Python package directory locally and upload it to the cluster.

### Create the Package Directory

```bash
mkdir -p zed_container_libs
```

### Install Required Dependencies

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

### Install the ZED Python Wrapper (Python 3.8)

```bash
pip install --target=./zed_container_libs \
    https://download.stereolabs.com/zedsdk/4.0/whl/linux_x86_64/pyzed-4.0-cp38-cp38-linux_x86_64.whl
```

Upload the complete `zed_container_libs` folder to your cluster home directory.

---

## Step 3 — Hardware Bridge & Authentication

### Extract USB Driver Bridge (Run Once on Cluster)

```bash
apptainer exec ~/zed_sdk.sif \
    cp /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 ~/libusb-1.0.so.0
```

This bridges the USB driver from the container to the host system.

### Google Cloud Credentials

To download DROID `.svo` files:

1. Place your service account key (e.g., `credentials.json`) in the project root.
2. The environment variable must point to it during execution:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
```

---

## Step 4 — Running with SLURM

Example SLURM script:

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




# 3. rdt_workspac 

The main project code is located in the `rdt_workspace` directory.

In the scripts folder contains: 
Here is the direct description of the files shown in your repository:

### Python Scripts

* **`evaluation.py`**: Runs the inference and success rate testing for the RDT model within the ManiSkill simulation.
* **`hdf5_maniskill_dataset.py`**: Defines the data structure and loading logic for ManiSkill trajectories stored in HDF5 files.
* **`prep_all.py`**: Executes the full data preparation sequence
* **`svo_online.py`**: Handles the direct extraction of depth data from ZED SVO files.
* **`unique_inst.py`**: Identifies and organizes unique natural language instructions to ensure the model trains on diverse tasks.

### Slurm and Shell Scripts

* **`finetune_maniskill.sh`**: The main cluster script for starting the RDT-DE fine-tuning process on ManiSkill.
* **`eval.sh`**: Submits the job to the cluster to run the `evaluation.py` script and record model performance.
* **`submit_prep.sh`**: Launches the `prep_all.py` job to handle data processing on cluster nodes.
* **`submit_droid.sh`**: Specifically handles the cluster submission for processing raw DROID dataset files.
* **`submit_instructions.sh`**: Submits jobs for processing or filtering the instruction set used for VLA training.



---

# Setup Summary

1. Create the Conda environment (`rdt_env`)
2. Pull the ZED Apptainer image
3. Create and upload the Python patch folder
4. Extract the USB bridge
5. Provide Google Cloud credentials
6. Run the SLURM job


