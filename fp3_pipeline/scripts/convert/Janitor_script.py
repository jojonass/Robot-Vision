import os
import shutil
from pathlib import Path
from tqdm import tqdm

def cleanup_empty_episodes(data_dir):
    data_path = Path(data_dir)
    # Get all immediate subdirectories
    episode_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    removed_count = 0
    print(f"Checking {len(episode_dirs)} folders for 'episode_data.h5'...")

    for folder in tqdm(episode_dirs):
        h5_file = folder / "episode_data.h5"
        
        if not h5_file.exists():
            # print(f"Deleting incomplete folder: {folder.name}")
            shutil.rmtree(folder)
            removed_count += 1

    print(f"--- Cleanup Complete ---")
    print(f"Folders removed: {removed_count}")
    print(f"Valid episodes remaining: {len(episode_dirs) - removed_count}")

if __name__ == "__main__":
    target_dir = "/home/e12434694/fp3_pipeline/data"
    cleanup_empty_episodes(target_dir)