Perfect — here is your **clean, readable, properly structured README**, ready to copy into `README.md`.

* Project name fixed: **RDT-DE**
* Environment name fixed: **rdt_env**
* Cleaner formatting (no unnecessary code blocks)
* Added section describing `rdt_workspace`

---

````markdown
# RDT-DE

RDT-DE provides the environment setup and cluster execution pipeline for running evaluation with the ZED SDK and DROID SVO trajectories on a SLURM-based GPU cluster.

The project uses:
- A Conda environment for Python dependencies
- An Apptainer container for CUDA and ZED SDK support
- A cluster patch strategy to bridge hardware and Python libraries

---

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

# 3. rdt_workspace Structure

The main project code is located in the `rdt_workspace` directory.

This directory contains:

* Core evaluation logic
* Model loading and inference code
* Simulation backend integration
* Scripts for trajectory processing
* ZED camera interaction through the Python wrapper
* Checkpoint loading and configuration handling

In particular:

* `scripts/evaluation.py`
  Entry point for running evaluation inside the container.

* `checkpoints/`
  Contains trained model weights.

* Configuration files
  Define simulation backend parameters and pretrained model paths.

The `rdt_workspace` directory is the main execution layer of the project.
All cluster jobs ultimately call code from this directory.

---

# Setup Summary

1. Create the Conda environment (`rdt_env`)
2. Pull the ZED Apptainer image
3. Create and upload the Python patch folder
4. Extract the USB bridge
5. Provide Google Cloud credentials
6. Run the SLURM job

The system is then ready for evaluation and data extraction.

```

---

If you’d like next, I can:

- Make this more concise and “publication style”
- Add a Quick Start section at the top
- Add a Troubleshooting section (recommended for cluster repos)
- Or make it more formal for a thesis/lab repository

Just tell me the target audience.
```
