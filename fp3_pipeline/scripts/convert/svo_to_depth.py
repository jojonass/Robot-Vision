import json
import os
import sys
import h5py
import cv2
import numpy as np
import functools
from pathlib import Path
from dataclasses import dataclass
import pyrallis
from google.cloud import storage

# Import your trusted utilities
from droid.postprocessing.util.svo2depth import export_depth
from droid.postprocessing.util.svo2mp4 import export_mp4

# --- THE NUCLEAR FLUSH ---
print = functools.partial(print, flush=True)
sys.stdout.reconfigure(line_buffering=False, write_through=True)

@dataclass
class OnlineDROIDConfig:
    bucket: str = "gresearch"
    prefix: str = "robotics/droid_raw/1.0.1/AUTOLab/success"
    data_dir: Path = Path("/home/e12434694/fp3_pipeline/raw_samples/data")
    temp_cache: Path = Path("/home/e12434694/temp_svo")
    stride: int = 3 
    limit: int = 10  # Smoke test limit

def pad_vector(arr, target_dim=128):
    if arr.shape[-1] >= target_dim: return arr[:, :target_dim]
    padding = np.zeros((arr.shape[0], target_dim - arr.shape[-1]))
    return np.hstack([arr, padding]).astype(np.float32)

def package_rdt_episode(trial_root, cam_serials, output_path, stride=3):
    """Consolidates processed MP4s and Depth H5s into RDT format."""
    images_dict, depths_dict = {}, {}
    cam_serials = sorted(cam_serials) # Ensure consistent cam order

    for i, serial in enumerate(cam_serials):
        mp4_path = trial_root / "recordings" / "MP4" / f"{serial}.mp4"
        depth_h5 = trial_root / f"{serial}_depth.h5"
        
        # RGB Processing
        cap = cv2.VideoCapture(str(mp4_path))
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % stride == 0:
                frame = cv2.resize(frame, (448, 448))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        cap.release()
        images_dict[f'cam_{i}'] = np.array(frames)
        
        # Depth Processing (Resize to match RGB)
        with h5py.File(depth_h5, 'r') as f_d:
            raw_depth = f_d['depth'][::stride]
            resized_depths = [cv2.resize(d, (448, 448), interpolation=cv2.INTER_NEAREST) for d in raw_depth]
            depths_dict[f'cam_{i}'] = np.array(resized_depths)

    # Load Trajectory
    with h5py.File(trial_root / "trajectory.h5", 'r') as f_t:
        raw_state = f_t['observations/robot_state'][::stride]
        raw_actions = f_t['action'][::stride]

    # Align Lengths
    L = min([len(v) for v in images_dict.values()] + [len(raw_state)])
    
    with h5py.File(output_path, 'w') as out:
        obs = out.create_group('observations')
        obs.create_dataset('images', data=np.stack([images_dict[f'cam_{i}'][:L] for i in range(len(cam_serials))], axis=1), compression="gzip")
        obs.create_dataset('depth', data=np.stack([depths_dict[f'cam_{i}'][:L] for i in range(len(cam_serials))], axis=1), compression="gzip")
        obs.create_dataset('state', data=pad_vector(raw_state[:L]))
        out.create_dataset('actions', data=pad_vector(raw_actions[:L]))
    return L

@pyrallis.wrap()
def postprocess(cfg: OnlineDROIDConfig):
    client = storage.Client()
    blobs = list(client.list_blobs(cfg.bucket, prefix=cfg.prefix))
    
    # Group SVOs by Trial
    trials = {}
    for b in [b for b in blobs if b.name.endswith(".svo")]:
        trial_id = "/".join(b.name.split("/")[-5:-3]) # date/trial
        if trial_id not in trials: trials[trial_id] = []
        trials[trial_id].append(b)

    processed_count = 0
    for trial_id, svo_blobs in trials.items():
        if processed_count >= cfg.limit: break
        
        trial_root = cfg.data_dir / trial_id
        rdt_output = trial_root / "rdt_episode.hdf5"
        if rdt_output.exists(): continue

        print(f"\n[*] Processing Trial: {trial_id}")
        cam_serials = []
        
        try:
            for blob in svo_blobs:
                cam_serial = Path(blob.name).stem
                cam_serials.append(cam_serial)
                local_svo = cfg.temp_cache / f"{cam_serial}.svo"
                
                # 1. Download & Extract using your blueprint's logic
                blob.download_to_filename(str(local_svo))
                export_depth(local_svo, trial_root / f"{cam_serial}_depth.h5")
                export_mp4(local_svo, trial_root / "recordings" / "MP4")
                local_svo.unlink()
                print(f"    + Extracted {cam_serial}")

            # 2. Trajectory Download
            traj_path = svo_blobs[0].name.split("recordings/")[0] + "trajectory.h5"
            client.bucket(cfg.bucket).blob(traj_path).download_to_filename(str(trial_root / "trajectory.h5"))

            # 3. RDT Packaging
            print(f"    -> Packaging RDT Episode...")
            package_rdt_episode(trial_root, cam_serials, rdt_output, stride=cfg.stride)
            
            # 4. Cleanup raw parts
            (trial_root / "trajectory.h5").unlink()
            for s in cam_serials:
                (trial_root / f"{s}_depth.h5").unlink()
                (trial_root / "recordings" / "MP4" / f"{s}.mp4").unlink()
            
            processed_count += 1
            print(f"    [SUCCESS] Trial Complete")

        except Exception as e:
            print(f"    [!] FAILED TRIAL {trial_id}: {e}")

if __name__ == "__main__":
    postprocess()