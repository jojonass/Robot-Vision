import os
import functools
import h5py
import numpy as np
import cv2
import json
import pickle
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from tqdm import tqdm
import pyzed.sl as sl
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
import pyrallis

# Force immediate log flushing for Slurm
print = functools.partial(print, flush=True)

@dataclass
class OnlineDROIDConfig:
    bucket: str = "gresearch"
    prefix: str = "robotics/droid_raw/1.0.1/AUTOLab/success"
    data_dir: Path = Path("/home/e12434694/fp3_pipeline/data")
    stats_dir: Path = Path("/home/e12434694/fp3_pipeline/stats")
    temp_cache: Path = Path("/home/e12434694/temp_svo")
    fp3_root: Path = Path("/home/e12434694/fp3_pipeline")
    limit: int = 10
    worker_id: int = int(os.environ.get("SLURM_PROCID", 0))
    num_workers: int = 8 

def extract_zed_frames(svo_path, settings_path, target_res=(256, 256)):
    serial = Path(svo_path).stem
    
    # Define rigid ranges for consistent contrast
    if serial.startswith("18"):
        MIN_DEPTH, MAX_DEPTH = 100.0, 600.0  # Wrist
    else:
        MIN_DEPTH, MAX_DEPTH = 300.0, 1100.0 # High

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.optional_settings_path = str(settings_path)

    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        return None, None, 0.0

    rgb_list, depth_list = [], []
    d_map, r_map = sl.Mat(), sl.Mat()
    out_h, out_w = target_res
    scale = out_h / 720.0
    tmp_w = int(1280 * scale)
    sx = (tmp_w - out_w) // 2

    try:
        while zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(d_map, sl.MEASURE.DEPTH)
            zed.retrieve_image(r_map, sl.VIEW.LEFT)

            d_res = cv2.resize(d_map.get_data(), (tmp_w, out_h), interpolation=cv2.INTER_NEAREST)
            d_crop = d_res[:, sx:sx + out_w]

            # Initialize with 0 (Black Background)
            d_final = np.zeros(d_crop.shape, dtype=np.uint8)
            valid_mask = (np.isfinite(d_crop)) & (d_crop >= MIN_DEPTH) & (d_crop <= MAX_DEPTH)

            if np.any(valid_mask):
                valid_data = d_crop[valid_mask]
                
                # --- HIGH CONTRAST NORMALIZATION ---
                # Maps [MIN, MAX] -> [255, 1]. Ensures objects at 30cm are pure white.
                denom = MAX_DEPTH - MIN_DEPTH
                normalized = 255 - (((valid_data - MIN_DEPTH) / denom) * 254)
                d_final[valid_mask] = np.clip(normalized, 1, 255).astype(np.uint8)

            r_res = cv2.resize(r_map.get_data()[:, :, :3], (tmp_w, out_h), interpolation=cv2.INTER_AREA)
            rgb_list.append(r_res[:, sx:sx + out_w])
            depth_list.append(d_final)
    finally:
        zed.close()
    
    return np.array(rgb_list), np.array(depth_list), 0.0

@pyrallis.wrap()
def postprocess(cfg: OnlineDROIDConfig):
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.stats_dir.mkdir(parents=True, exist_ok=True)
    cfg.temp_cache.mkdir(parents=True, exist_ok=True)
    
    worker_log = {"instructions": [], "state_stats": []}
    instruction_counts = Counter()
    all_instructions_txt = []

    client = storage.Client(project="anonymous", credentials=AnonymousCredentials())
    blobs = list(client.list_blobs(cfg.bucket, prefix=cfg.prefix))

    trials = {}
    for b in blobs:
        parts = b.name.split("/")
        try: tid = parts[parts.index("success") + 2]
        except: continue
        trials.setdefault(tid, {"svos": [], "trajectory": None, "metadata_json": None})
        if "trajectory.h5" in b.name: trials[tid]["trajectory"] = b
        elif "metadata_" in b.name and b.name.endswith(".json"): trials[tid]["metadata_json"] = b
        elif b.name.lower().endswith(".svo"): trials[tid]["svos"].append(b)

    valid_keys = sorted(k for k, v in trials.items() if v["trajectory"] and len(v["svos"]) >= 3)
    my_slice = [tid for i, tid in enumerate(valid_keys) if i % cfg.num_workers == cfg.worker_id][:cfg.limit]

    print(f"Worker {cfg.worker_id} starting {len(my_slice)} trials.")

    for tid in tqdm(my_slice):
        safe_tid = tid.replace(":", "_")
        trial_root = cfg.data_dir / safe_tid
        final_h5 = trial_root / "episode_data.h5"
        if final_h5.exists(): continue
        
        trial_root.mkdir(parents=True, exist_ok=True)
        local_garbage = []

        # 1. Extract Instruction
        instruction = "Perform the task"
        if trials[tid]["metadata_json"]:
            l_meta = cfg.temp_cache / f"w{cfg.worker_id}_{safe_tid}_meta.json"
            trials[tid]["metadata_json"].download_to_filename(str(l_meta))
            with open(l_meta) as f:
                instruction = json.load(f).get("current_task", instruction)
            local_garbage.append(l_meta)
        
        # Add to the text log list
        all_instructions_txt.append(f"{tid} | {instruction}")
        instruction_counts[instruction] += 1
        worker_log["instructions"].append({"tid": tid, "instruction": instruction})

        # 2. Extract Trajectory
        l_traj = cfg.temp_cache / f"w{cfg.worker_id}_{safe_tid}_traj.h5"
        trials[tid]["trajectory"].download_to_filename(str(l_traj))
        local_garbage.append(l_traj)
        with h5py.File(l_traj, "r") as f:
            base = "observation/robot_state"
            qpos, qvel = f[f"{base}/joint_positions"][:], f[f"{base}/joint_velocities"][:]
            cart, grip = f[f"{base}/cartesian_position"][:], f[f"{base}/gripper_position"][:]
            num_frames = len(qpos)

        # 3. Video Processing
        cam_data = {}
        view_slots = ["cam_high", "cam_left_wrist"]
        v_idx = 0
        for b in sorted(trials[tid]["svos"], key=lambda x: x.name):
            serial = Path(b.name).stem
            slot = "cam_right_wrist" if serial.startswith("18") else (view_slots[v_idx] if v_idx < len(view_slots) else None)
            if slot:
                l_svo = cfg.temp_cache / f"w{cfg.worker_id}_{safe_tid}_{serial}.svo"
                b.download_to_filename(str(l_svo))
                local_garbage.append(l_svo)
                rgb, depth, _ = extract_zed_frames(l_svo, cfg.temp_cache)
                if rgb is not None:
                    idx = np.linspace(0, len(rgb) - 1, num_frames).astype(int)
                    cam_data[slot], cam_data[f"{slot}_depth"] = rgb[idx], depth[idx]
                if not serial.startswith("18"): v_idx += 1

        # 4. Save to HDF5
        if all(k in cam_data for k in ["cam_high", "cam_left_wrist", "cam_right_wrist"]):
            with h5py.File(final_h5, "w") as f_out:
                rdt = np.zeros((num_frames, 128), dtype=np.float32)
                rdt[:, 0:7], rdt[:, 10:17] = qpos, qvel
                rdt[:, 20:26], rdt[:, 29] = cart[:, :6], grip.flatten()
                worker_log["state_stats"].append({"tid": tid, "mean": np.mean(rdt, axis=0)})
                f_out.create_dataset("state", data=rdt, compression="lzf")
                for k, v in cam_data.items():
                    f_out.create_dataset(k, data=v, compression="gzip", compression_opts=4)
                f_out.attrs["language_instruction"] = np.string_(instruction)

        for p in local_garbage: 
            if p.exists(): p.unlink()

    # --- THE TEXT FILE STUFF (FINAL OUTPUTS) ---
    # 1. Raw Text log (ID | Instruction)
    with open(cfg.fp3_root / f"instructions_worker_{cfg.worker_id}.txt", "w") as f_txt:
        f_txt.write("\n".join(all_instructions_txt))
    
    # 2. Audit Pickle
    with open(cfg.stats_dir / f"audit_worker_{cfg.worker_id}.pkl", "wb") as f_audit:
        pickle.dump(worker_log, f_audit)
    
    # 3. Task Inventory JSON
    with open(cfg.stats_dir / f"task_inventory_worker_{cfg.worker_id}.json", "w") as f_task:
        json.dump(dict(instruction_counts), f_task, indent=4)

if __name__ == "__main__":
    postprocess()