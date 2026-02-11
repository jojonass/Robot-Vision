

---

````markdown
# RDT-DE

This repository contains the environment setup and cluster execution pipeline for running RDT-DE with the ZED SDK and DROID SVO evaluation on a SLURM-based cluster.

---

# 1. Environment Setup (Conda)

To ensure reproducibility, recreate the Python environment using the provided `environment.yml` file.

### Create the Environment

```bash
conda env create -f environment.yml
````

### Activate the Environment

```bash
conda activate <environment_name>
```

Replace `<environment_name>` with the name specified inside `environment.yml`.

### (Optional) Verify Installation

```bash
python --version
conda list
```

---

# 2. Data Extraction Pipeline (Apptainer + ZED SDK)

The data extraction pipeline runs inside an Apptainer (Singularity) container due to cluster restrictions.

We use a **Cluster Patch strategy** to bridge ZED hardware, CUDA, and external Python dependencies.

---

## Step 1 — Pull the Base Apptainer Image

The container provides:

* Ubuntu 20.04
* CUDA 11.4
* ZED C++ SDK
* OpenGL/EGL support for off-screen rendering

Pull the image on the cluster:

```bash
apptainer pull ~/zed_sdk.sif docker://stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04
```

The `gl-devel` version is required for EGL-based rendering.

---

## Step 2 — Prepare the Python "Cluster Patch"

Since the `.sif` image is read-only, create a portable Python package folder locally and upload it to the cluster.

### 2.1 Create Local Package Bundle

Run on a local machine with internet access:

```bash
mkdir -p zed_container_libs

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

### 2.2 Install the ZED Python Wrapper (Python 3.8)

```bash
pip install --target=./zed_container_libs \
    https://download.stereolabs.com/zedsdk/4.0/whl/linux_x86_64/pyzed-4.0-cp38-cp38-linux_x86_64.whl
```

Upload the entire `zed_container_libs` folder to your cluster home directory.

---

## Step 3 — Hardware Bridge & Authentication

### 3.1 Extract USB Driver Bridge (Run Once on Cluster)

```bash
apptainer exec ~/zed_sdk.sif \
    cp /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 ~/libusb-1.0.so.0
```

### 3.2 Google Cloud Credentials

To download DROID `.svo` files:

1. Place your Service Account key (e.g., `credentials.json`) in the project root.
2. The environment will use this file for authentication.

---

## Step 4 — Running via SLURM

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

# Setup Summary

1. Create the Conda environment
2. Pull the ZED Apptainer image
3. Create and upload the Python patch folder
4. Extract the USB bridge
5. Add Google Cloud credentials
6. Run the SLURM job

```

---

If you later want a shorter “Quick Start” version for external collaborators, I can compress this into a one-page minimal setup guide.
```
