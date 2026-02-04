import os
import h5py
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

def compute_dataset_stats(data_dir, output_path):
    data_path = Path(data_dir)
    # Only look for files in folders that passed your Janitor cleanup
    h5_files = list(data_path.rglob('episode_data.h5'))
    
    all_states = []
    all_actions = []

    print(f"📊 Analyzing {len(h5_files)} episodes for statistics...")
    
    for h5_path in tqdm(h5_files):
        try:
            with h5py.File(h5_path, 'r') as f:
                # These are already the 128-dim vectors
                all_states.append(f['state'][:])
                all_actions.append(f['action'][:])
        except Exception as e:
            print(f"⚠️ Error reading {h5_path}: {e}")

    # Stack all frames from all episodes
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    # Compute global stats
    stats = {
        'state': {
            'mean': np.mean(all_states, axis=0),
            'std': np.std(all_states, axis=0),
            'min': np.min(all_states, axis=0),
            'max': np.max(all_states, axis=0),
        },
        'action': {
            'mean': np.mean(all_actions, axis=0),
            'std': np.std(all_actions, axis=0),
            'min': np.min(all_actions, axis=0),
            'max': np.max(all_actions, axis=0),
        }
    }

    # Save to the root directory
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"✅ Success! stats.pkl saved to {output_path}")

if __name__ == "__main__":
    # Update these paths to your environment
    DATA_DIR = "/home/e12434694/fp3_pipeline/data"
    STATS_OUTPUT = "/home/e12434694/fp3_pipeline/stats.pkl"
    
    compute_dataset_stats(DATA_DIR, STATS_OUTPUT)