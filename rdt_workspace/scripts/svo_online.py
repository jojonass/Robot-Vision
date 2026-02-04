import os
import functools
import h5py
import numpy as np
import cv2
import json
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import pyzed.sl as sl
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
import pyrallis

# Force immediate log flushing for Slurm logs
print = functools.partial(print, flush=True)

@dataclass
class OnlineDROIDConfig:
    bucket: str = "gresearch"
    prefix: str = "robotics/droid_raw/1.0.1/AUTOLab/success"
    workspace_root: Path = Path("/home/e12434694/rdt_workspace")
    data_dir: Path = Path("/home/e12434694/rdt_workspace/test_data/hdf5")
    temp_cache: Path = Path("/home/e12434694/temp_svo")
    worker_id: int = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) 
    num_workers: int = 8 
    target_hz: int = 25 
    samples_per_category: int = 2  # This is now a GLOBAL limit

TASK_WHITELIST = {
    "drop yellow block into blue cup": "pick_cube",
    "stack cup on bowl": "stack_cube",
    "stack two cups on each other": "stack_cube",
    "stack cups": "stack_cube",
    "stack 2 bowls on top of one another": "stack_cube",
    "stack 4 cups": "stack_cube",
    "stack 3 cups": "stack_cube",
    "stack 3 cups together": "stack_cube",
    "stack blocks": "stack_cube",
    "take disk off peg": "peg_insertion",
    "place disk on peg": "peg_insertion",
    "insert red and green block into black bowl": "peg_insertion",
    "insert plug into socket": "plug_charger",
    "insert socket into plug": "plug_charger",
    "move object to a new position and orientation ex grasping relocating flipping": "push_cube"
}

# [extract_zed_frames remains identical to your working version]
def extract_zed_frames(svo_path, settings_path, slot, target_res=(384, 384)):
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.optional_settings_path = str(settings_path)
    init_params.svo_real_time_mode = False
    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS: return None, None
    runtime_scan = sl.RuntimeParameters()
    point_cloud = sl.Mat()
    frame_ranges = []
    while zed.grab(runtime_scan) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        pc_data = point_cloud.get_data()
        euclidean_dist = np.sqrt(pc_data[:,:,0]**2 + pc_data[:,:,1]**2 + pc_data[:,:,2]**2)
        valid = np.isfinite(euclidean_dist)
        if np.sum(valid) > 100:
            p5, p95 = np.percentile(euclidean_dist[valid], 5), np.percentile(euclidean_dist[valid], 95)
            frame_ranges.append((p5, p95, p95 - p5))
    if frame_ranges:
        min_range_idx = np.argmin([r[2] for r in frame_ranges]); t_min, t_max, t_range = frame_ranges[min_range_idx]
        t_range = max(t_range, 200 if "wrist" in slot.lower() else 350)
    else: t_min, t_max, t_range = 200, 800, 600
    zed.set_svo_position(0)
    rgb_list, depth_list, r_map, pc_mat = [], [], sl.Mat(), sl.Mat()
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    while zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(r_map, sl.VIEW.LEFT); zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
        raw_rgb = r_map.get_data()[:, :, :3].copy(); pc_data = pc_mat.get_data()
        dist = np.sqrt(pc_data[:,:,0]**2 + pc_data[:,:,1]**2 + pc_data[:,:,2]**2)
        norm = np.clip((dist - t_min) / t_range, 0, 1)
        d_gray = (norm * 255).astype(np.uint8); d_gray = clahe.apply(d_gray)
        d_color = cv2.applyColorMap(d_gray, cv2.COLORMAP_JET)
        canny = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY), (3,3), 0), 80, 160)
        d_color[canny > 0] = [255, 255, 255]
        H, W = raw_rgb.shape[:2]; x_start = (W - H) // 2
        rgb_list.append(cv2.resize(raw_rgb[:, x_start:x_start+H], target_res, interpolation=cv2.INTER_AREA))
        depth_list.append(cv2.resize(d_color[:, x_start:x_start+H], target_res, interpolation=cv2.INTER_NEAREST))
    zed.close()
    return np.array(rgb_list), np.array(depth_list)

def get_global_count(data_dir, category):
    """Checks the shared filesystem to see how many of this category are done."""
    count = 0
    # We look for files that have the category saved in their metadata attributes
    # Or more efficiently, check if you've implemented a naming convention.
    # For speed, we will count files that have been successfully finalized.
    for p in data_dir.glob("*.hdf5"):
        try:
            with h5py.File(p, 'r') as f:
                if f.attrs.get("task_category", b"").decode() == category:
                    count += 1
        except: continue
    return count

@pyrallis.wrap()
def postprocess(cfg: OnlineDROIDConfig):
    for d in [cfg.data_dir, cfg.temp_cache]: d.mkdir(parents=True, exist_ok=True)
    
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

    # Distributed Search Slice
    all_keys = sorted(k for k, v in trials.items() if v["trajectory"] and len(v["svos"]) >= 2)
    my_search_slice = [tid for i, tid in enumerate(all_keys) if i % cfg.num_workers == cfg.worker_id]
    
    # Pre-scan metadata locally
    my_candidate_trials = []
    for tid in tqdm(my_search_slice, desc=f"W{cfg.worker_id} Scanning"):
        if not trials[tid]["metadata_json"]: continue
        safe_tid = tid.replace(':', '_')
        l_meta = cfg.temp_cache / f"worker{cfg.worker_id}_meta_{safe_tid}.json"
        try:
            trials[tid]["metadata_json"].download_to_filename(str(l_meta))
            with open(l_meta) as f:
                instr = json.load(f).get("current_task", "").lower().strip()
                if instr in TASK_WHITELIST:
                    trials[tid].update({"cat": TASK_WHITELIST[instr], "instr": instr})
                    my_candidate_trials.append(tid)
        except: pass
        finally:
            if l_meta.exists(): l_meta.unlink()

    # --- GLOBAL-AWARE EXTRACTION LOOP ---
    for tid in tqdm(my_candidate_trials, desc=f"Worker {cfg.worker_id} Processing"):
        cat = trials[tid]["cat"]
        
        # Check global count before starting heavy SVO download
        current_global_count = get_global_count(cfg.data_dir, cat)
        if current_global_count >= cfg.samples_per_category:
            continue # Skip! Global limit reached for this category
            
        safe_tid = tid.replace(":", "_")
        final_h5 = cfg.data_dir / f"{safe_tid}.hdf5"
        if final_h5.exists(): continue

        # [Heavy processing starts here...]
        local_garbage = []
        l_traj = cfg.temp_cache / f"w{cfg.worker_id}_{safe_tid}_traj.h5"
        trials[tid]["trajectory"].download_to_filename(str(l_traj))
        local_garbage.append(l_traj)
        
        with h5py.File(l_traj, "r") as f:
            base = "observation/robot_state"
            qpos_raw, qvel_raw = f[f"{base}/joint_positions"][:], f[f"{base}/joint_velocities"][:]
            cart_raw, grip_raw = f[f"{base}/cartesian_position"][:], f[f"{base}/gripper_position"][:]
            num_target_steps = int((len(qpos_raw) / 15.0) * cfg.target_hz)

        cam_data = {}
        view_slots = ["cam_high", "cam_left"]
        v_idx = 0
        for b in sorted(trials[tid]["svos"], key=lambda x: x.name):
            serial = Path(b.name).stem
            slot = "cam_right_wrist" if serial.startswith("18") else (view_slots[v_idx] if v_idx < 2 else None)
            if slot:
                l_svo = cfg.temp_cache / f"w{cfg.worker_id}_{safe_tid}_{serial}.svo"
                b.download_to_filename(str(l_svo))
                local_garbage.append(l_svo)
                rgb, depth = extract_zed_frames(l_svo, cfg.temp_cache, slot)
                if rgb is not None:
                    idx = np.linspace(0, len(rgb)-1, num_target_steps).astype(int)
                    cam_data[slot], cam_data[f"{slot}_depth"] = rgb[idx], depth[idx]
                if not serial.startswith("18"): v_idx += 1

        if all(k in cam_data for k in ["cam_high", "cam_right_wrist"]):
            l_img = cam_data.get("cam_left", np.zeros_like(cam_data["cam_high"]))
            l_dep = cam_data.get("cam_left_depth", np.zeros_like(cam_data["cam_high_depth"]))
            full_img = np.concatenate([cam_data["cam_high"], l_img, cam_data["cam_right_wrist"]], axis=2)
            full_dep = np.concatenate([cam_data["cam_high_depth"], l_dep, cam_data["cam_right_wrist_depth"]], axis=2)
            
            idx_r = np.linspace(0, len(qpos_raw)-1, num_target_steps).astype(int)
            with h5py.File(final_h5, "w") as f_out:
                state = np.zeros((num_target_steps, 128), dtype=np.float32)
                state[:, 0:7], state[:, 10:17] = qpos_raw[idx_r], qvel_raw[idx_r]
                state[:, 20:26], state[:, 29] = cart_raw[idx_r, :6], grip_raw[idx_r].reshape(-1)
                f_out.create_dataset("state", data=state, compression="lzf")
                f_out.create_dataset("image", data=full_img, compression="lzf")
                f_out.create_dataset("depth", data=full_dep, compression="lzf")
                f_out.attrs["language_instruction"] = np.string_(trials[tid]["instr"])
                f_out.attrs["task_category"] = np.string_(cat)
            print(f"  [SUCCESS] Global Count for {cat}: {get_global_count(cfg.data_dir, cat)}")

        for p in local_garbage: 
            if p.exists(): p.unlink()

if __name__ == "__main__":
    postprocess()