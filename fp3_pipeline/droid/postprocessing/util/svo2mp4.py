"""
svo2mp4.py
Utility scripts for using the ZED Python SDK to convert raw `.svo` files to `.mp4` files.
This version uses native ZED SDK recording to avoid FFMPEG/OpenCV dependencies.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pyzed.sl as sl
from tqdm import tqdm



import os
import cv2
import pyzed.sl as sl
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import pyzed.sl as sl
from pathlib import Path
from PIL import Image  # Standard Python Imaging Library

def export_mp4(svo_file: Path, mp4_dir: Path, stereo_view: str = "left", show_progress: bool = False) -> bool:
    """Extracts SVO frames using PIL to bypass OpenCV/system library failures."""
    frames_out_dir = mp4_dir / svo_file.stem
    os.makedirs(frames_out_dir, exist_ok=True)
    
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_file))
    init_params.svo_real_time_mode = False

    init_params.camera_disable_self_calib = True  # Disables the need for online calib
    init_params.optional_settings_path = "/home/e12434694/zed_settings"
    
    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        return False

    n_frames = zed.get_svo_number_of_frames()
    rt_params = sl.RuntimeParameters()
    img_container = sl.Mat()
    view = sl.VIEW.LEFT if stereo_view == "left" else sl.VIEW.RIGHT

    success_count = 0
    try:
        for i in range(n_frames):
            if zed.grab(rt_params) == sl.ERROR_CODE.SUCCESS:
                # Force retrieve to CPU
                zed.retrieve_image(img_container, view, sl.MEM.CPU)
                
                # Get the data and convert from RGBA to RGB
                # ZED's RGBA format is (H, W, 4)
                rgba_data = img_container.get_data()
                if rgba_data is None: continue
                
                # Slicing [..., :3] removes the alpha channel
                # [..., ::-1] converts BGRA to RGB if needed
                rgb_data = rgba_data[:, :, :3][:, :, ::-1] 
                
                # Use PIL to save
                img_path = frames_out_dir / f"{i:06d}.png"
                img = Image.fromarray(rgb_data)
                img.save(str(img_path))
                
                success_count += 1
            else:
                break
    except Exception as e:
        print(f"\n[!] PIL Extraction Error: {e}", flush=True)
        return False
    finally:
        zed.close()

    return success_count > 0

# --- Update your record_paths in the convert_mp4s function as well ---
# Make sure the paths point to the FOLDERS instead of .mp4 files

def convert_mp4s(
    data_dir: Path,
    demo_dir: Path,
    wrist_serial: str,
    ext1_serial: str,
    ext2_serial: str,
    ext1_extrinsics: List[float],
    ext2_extrinsics: List[float],
    do_fuse: bool = False,
) -> Tuple[bool, Optional[Dict[str, str]]]:
    """Orchestrates the conversion of all SVOs in a trial directory."""
    svo_path = demo_dir / "recordings" / "SVO"
    mp4_path = demo_dir / "recordings" / "MP4"
    os.makedirs(mp4_path, exist_ok=True)
    
    for svo_file in svo_path.glob("*.svo"):
        success = export_mp4(svo_file, mp4_path, show_progress=True)
        if not success:
            return False, None

    # Identify camera positions for downstream tracking
    ext1_y, ext2_y = ext1_extrinsics[1], ext2_extrinsics[1]
    left_serial = ext1_serial if ext1_y > ext2_y else ext2_serial
    right_serial = ext2_serial if left_serial == ext1_serial else ext1_serial

    rel_svo_path = svo_path.relative_to(data_dir)
    rel_mp4_path = mp4_path.relative_to(data_dir)
    
    record_paths = {
        "wrist_svo_path": str(rel_svo_path / f"{wrist_serial}.svo"),
        "wrist_mp4_path": str(rel_mp4_path / f"{wrist_serial}.mp4"),
        "ext1_svo_path": str(rel_svo_path / f"{ext1_serial}.svo"),
        "ext1_mp4_path": str(rel_mp4_path / f"{ext1_serial}.mp4"),
        "ext2_svo_path": str(rel_svo_path / f"{ext2_serial}.svo"),
        "ext2_mp4_path": str(rel_mp4_path / f"{ext2_serial}.mp4"),
        "left_mp4_path": str(rel_mp4_path / f"{left_serial}.mp4"),
        "right_mp4_path": str(rel_mp4_path / f"{right_serial}.mp4"),
    }

    # NOTE: do_fuse is disabled here because 'ffmpeg' is missing from the container.
    if do_fuse:
        print("    [!] Warning: 'do_fuse' requested but ffmpeg is missing. Skipping fusion.")

    return True, record_paths