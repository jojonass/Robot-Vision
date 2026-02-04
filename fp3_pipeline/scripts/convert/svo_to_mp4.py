import os
import sys
import h5py
import cv2
import shutil
import numpy as np
import functools
from pathlib import Path
from dataclasses import dataclass
import pyrallis
from google.cloud import storage

# DROID utils
from droid.postprocessing.util.svo2depth import export_depth
from droid.postprocessing.util.svo2mp4 import export_mp4

# Flush everything for SLURM
print = functools.partial(print, flush=True)
sys.stdout.reconfigure(line_buffering=False, write_through=True)


@dataclass
class OnlineDROIDConfig:
    bucket: str = "gresearch"
    prefix: str = "robotics/droid_raw/1.0.1/AUTOLab/success"
    data_dir: Path = Path("/home/e12434694/fp3_pipeline/raw_samples/data")
    temp_cache: Path = Path("/home/e12434694/temp_svo")
    stride: int = 3
    limit: int = 10
    min_disk_gb: int = 1


def get_free_space_gb(path: str) -> float:
    stat = os.statvfs(path)
    return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)


def pad_vector(arr: np.ndarray, target_dim=128) -> np.ndarray:
    if arr.shape[-1] >= target_dim:
        return arr.astype(np.float32)
    padding = np.zeros((arr.shape[0], target_dim - arr.shape[-1]))
    return np.hstack([arr, padding]).astype(np.float32)


def clean_path(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def package_rdt_episode(trial_root: Path, cam_serials: list, output_path: Path, stride=3):
    """Consolidates RGB MP4s + depth into an RDT HDF5 file."""
    images_dict, depths_dict = {}, {}

    for i, serial in enumerate(sorted(cam_serials)):
        mp4_path = trial_root / "recordings" / "MP4" / f"{serial}.mp4"
        depth_h5 = trial_root / f"{serial}_depth.h5"

        # RGB frames
        cap = cv2.VideoCapture(str(mp4_path))
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % stride == 0:
                frame = cv2.resize(frame, (448, 448))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        cap.release()
        images_dict[f"cam_{i}"] = np.array(frames)

        # Depth
        with h5py.File(depth_h5, "r") as f_d:
            d_key = "depth" if "depth" in f_d else list(f_d.keys())[0]
            raw_depth = f_d[d_key][::stride]
            depths_dict[f"cam_{i}"] = np.array(
                [cv2.resize(d, (448, 448), interpolation=cv2.INTER_NEAREST) for d in raw_depth]
            )

    # Trajectory
    with h5py.File(trial_root / "trajectory.h5", "r") as f_t:
        raw_state = f_t["observations/robot_state"][::stride]
        raw_actions = f_t["action"][::stride]

    # Align lengths
    L = min([len(v) for v in images_dict.values()] + [len(raw_state)])

    with h5py.File(output_path, "w") as out:
        obs = out.create_group("observations")
        obs.create_dataset(
            "images",
            data=np.stack([images_dict[f"cam_{i}"][:L] for i in range(len(cam_serials))], axis=1),
            compression="gzip",
        )
        obs.create_dataset(
            "depth",
            data=np.stack([depths_dict[f"cam_{i}"][:L] for i in range(len(cam_serials))], axis=1),
            compression="gzip",
        )
        obs.create_dataset("state", data=pad_vector(raw_state[:L]))
        out.create_dataset("actions", data=pad_vector(raw_actions[:L]))
    return L


@pyrallis.wrap()
def postprocess(cfg: OnlineDROIDConfig):
    client = storage.Client()
    blobs = list(client.list_blobs(cfg.bucket, prefix=cfg.prefix))
    cfg.temp_cache.mkdir(parents=True, exist_ok=True)

    # Group SVOs by trial
    trials = {}
    for b in [b for b in blobs if b.name.endswith(".svo")]:
        trial_id = "/".join(b.name.split("/")[-5:-3])
        trials.setdefault(trial_id, []).append(b)

    for trial_id, svo_blobs in trials.items():
        if get_free_space_gb(str(cfg.data_dir)) < cfg.min_disk_gb:
            print("!!! DISK SPACE LOW. STOPPING.")
            break

        trial_root = cfg.data_dir / trial_id
        rdt_out = trial_root / "rdt_episode.hdf5"
        if rdt_out.exists():
            continue

        print(f"\n[*] Processing Trial: {trial_id}")
        (trial_root / "recordings" / "MP4").mkdir(parents=True, exist_ok=True)
        cam_serials = []

        
        try:
            for blob in svo_blobs:
                
                cam_serial = os.path.basename(blob.name).replace(".svo", "")
                print('wtf I am working')
                cam_serials.append(cam_serial)
                local_svo = cfg.temp_cache / f"{cam_serial}.svo"
                depth_file = trial_root / f"{cam_serial}_depth.h5"

                clean_path(depth_file)

                # Download & convert
                blob.download_to_filename(str(local_svo))
                export_depth(local_svo, depth_file)
                export_mp4(local_svo, trial_root / "recordings" / "MP4")
                local_svo.unlink()
                print(f"    + Extracted {cam_serial}")

            # Trajectory
            traj_blob_path = svo_blobs[0].name.split("recordings/")[0] + "trajectory.h5"
            client.bucket(cfg.bucket).blob(traj_blob_path).download_to_filename(str(trial_root / "trajectory.h5"))

            # Package RDT
            print("    -> Packaging RDT Episode...")
            steps = package_rdt_episode(trial_root, cam_serials, rdt_out, cfg.stride)
            print(f"    [SUCCESS] Trial Complete | Steps: {steps}")

            # Cleanup
            (trial_root / "trajectory.h5").unlink()
            for s in cam_serials:
                clean_path(trial_root / f"{s}_depth.h5")
                clean_path(trial_root / "recordings" / "MP4" / f"{s}.mp4")

        except Exception as e:
            import traceback
            print(f"    [!] FAILED TRIAL {trial_id}")
            traceback.print_exc()
            # Clean temp SVOs
            for s in cam_serials:
                temp_svo = cfg.temp_cache / f"{s}.svo"
                if temp_svo.exists():
                    temp_svo.unlink()


if __name__ == "__main__":
    postprocess()
