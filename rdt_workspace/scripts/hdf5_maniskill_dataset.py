import os
import shutil
import h5py
import numpy as np
import yaml
import torch
from pathlib import Path
from scipy.interpolate import interp1d

# --- DISTRIBUTED-SAFE FRESH START ---
def cleanup_pycache():
    # Only the master process should handle cleanup to avoid race conditions
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        curr_dir = Path(__file__).parent
        pycache = curr_dir / "__pycache__"
        if pycache.exists():
            try:
                shutil.rmtree(pycache)
                print(f"[DEBUG] Master Process: Cleared {pycache}")
            except OSError:
                pass

cleanup_pycache()

def interpolate_action_sequence(action_sequence, target_size):
    N, D = action_sequence.shape
    indices_old = np.arange(N)
    indices_new = np.linspace(0, N - 1, target_size)
    interp_func = interp1d(indices_old, action_sequence, kind='linear', axis=0, assume_sorted=True)
    return interp_func(indices_new)

class HDF5VLADataset:
    def __init__(self, config_path="/home/e12434694/rdt_workspace/scripts/configs/base.yaml"):
        print(f"\n--- Initializing HDF5 Loader ---")
        print(f"Reading config from: {config_path}")
        
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        self.IMG_HISTORY_SIZE = cfg["common"].get("img_history_size", 2)
        self.STATE_DIM = cfg["common"].get("state_dim", 128)
        self.CHUNK_SIZE = cfg["common"].get("action_chunk_size", 64)
        
        self.data_dir = Path(cfg["dataset"]["data_dir"])
        self.embed_dir = Path(cfg["dataset"]["lang_embed_path"])
        self.DATASET_NAME = "maniskill_droid_v1"

        print(f"Checking HDF5 Directory: {self.data_dir}")
        print(f"Checking Embed Directory: {self.embed_dir}")

        all_h5 = sorted(list(self.data_dir.glob('*.hdf5')))
        print(f"Found {len(all_h5)} total .hdf5 files.")

        self.episode_paths = []
        for p in all_h5:
            # Match by stem (e.g., Fri_Aug_18_11_40_54_2023)
            expected_npy = self.embed_dir / f"{p.stem}.npy"
            if expected_npy.exists():
                self.episode_paths.append(p)
            else:
                print(f"[MISSING] No match for: {p.name}")
                print(f"          Expected at: {expected_npy}")

        print(f"Final Count: {len(self.episode_paths)} episodes ready for training.")
        print(f"---------------------------------\n")

    def __len__(self):
        return len(self.episode_paths)

    def __getitem__(self, index):
        valid, sample = self.parse_hdf5_file(index)
        if not valid:
            # Avoid getting stuck; try a random index if one fails
            new_idx = np.random.randint(0, len(self.episode_paths))
            return self.__getitem__(new_idx)
        return sample

    def parse_hdf5_file(self, index):
        h5_path = self.episode_paths[index]
        npy_path = self.embed_dir / f"{h5_path.stem}.npy"
        
        try:
            # Load T5 Language Embedding
            lang_embed = torch.from_numpy(np.load(str(npy_path))).float()
            
            # Ensure embedding is (1, Token_Len, 4096) or compatible with RDT
            # If your precompute script saved (4096,), we unsqueeze it
            if lang_embed.ndim == 1:
                lang_embed = lang_embed.unsqueeze(0)

            with h5py.File(str(h5_path), 'r') as f:
                instruction = f.attrs.get('language_instruction', "perform task")
                if isinstance(instruction, bytes): instruction = instruction.decode('utf-8')

                states = f['state'][:] 
                num_steps = states.shape[0]
                
                # Sample a step, ensuring enough buffer for history and future actions
                # We use 16 steps of future data to interpolate into the 64-step chunk
                step_index = np.random.randint(self.IMG_HISTORY_SIZE, max(self.IMG_HISTORY_SIZE + 1, num_steps - 20))

                # --- 1. Images Processing (25Hz Rainbow) ---
                full_rgb = f['image'][:]    
                start_idx = max(0, step_index - self.IMG_HISTORY_SIZE + 1)
                rgb_seq = full_rgb[start_idx : step_index + 1]
                
                # Padding for sequence start
                if rgb_seq.shape[0] < self.IMG_HISTORY_SIZE:
                    pad = self.IMG_HISTORY_SIZE - rgb_seq.shape[0]
                    rgb_seq = np.concatenate([np.tile(rgb_seq[0:1], (pad, 1, 1, 1)), rgb_seq], axis=0)

                # RDT Unrolling: [Cam1_T-1, Cam1_T0, Cam2_T-1, Cam2_T0, Cam3_T-1, Cam3_T0]
                image_list = []
                for cam_idx in range(3): 
                    # Extract 384x384 crop for current camera from the wide strip
                    strip = rgb_seq[:, :, cam_idx*384 : (cam_idx+1)*384, :]
                    # Normalize and convert to (T, C, H, W)
                    tensors = torch.from_numpy(strip.transpose(0, 3, 1, 2)).float() / 255.0
                    for t in range(self.IMG_HISTORY_SIZE):
                        image_list.append(tensors[t])

                # --- 2. States & Actions ---
                curr_state = states[step_index]
                padded_state = np.zeros((self.STATE_DIM,), dtype=np.float32)
                padded_state[:min(len(curr_state), self.STATE_DIM)] = curr_state[:self.STATE_DIM]

                # Grab 16 future steps and interpolate to 64
                raw_actions = states[step_index : step_index + 16]
                if len(raw_actions) < 16:
                    pad_len = 16 - len(raw_actions)
                    raw_actions = np.concatenate([raw_actions, np.tile(raw_actions[-1:], (pad_len, 1))], axis=0)
                
                full_actions = np.zeros((16, self.STATE_DIM), dtype=np.float32)
                full_actions[:, :min(raw_actions.shape[1], self.STATE_DIM)] = raw_actions[:, :self.STATE_DIM]
                action_seq = interpolate_action_sequence(full_actions, self.CHUNK_SIZE)

                return True, {
                    "meta": {"dataset_name": self.DATASET_NAME, "instruction": instruction},
                    "states": torch.from_numpy(padded_state).unsqueeze(0), # (1, 128)
                    "actions": torch.from_numpy(action_seq).float(), # (64, 128)
                    "state_elem_mask": torch.ones(self.STATE_DIM),
                    "state_norm": torch.ones(self.STATE_DIM),
                    "lang_embed": lang_embed,
                    "images": image_list, 
                    "ctrl_freq": torch.tensor([25.0]),
                    "data_idx": index
                }
        except Exception as e:
            print(f"[ERROR] Parsing {h5_path.name}: {e}")
            return False, None

# --- TEST BLOCK ---
if __name__ == "__main__":
    try:
        loader = HDF5VLADataset()
        if len(loader) > 0:
            sample = loader.__getitem__(0)
            print("Successfully extracted sample!")
            print(f" - Instruction: {sample['meta']['instruction']}")
            print(f" - Image List Len: {len(sample['images'])} (Expected 6 for 3 cams * 2 history)")
            print(f" - Action Shape: {sample['actions'].shape} (Expected 64x128)")
            print(f" - Ctrl Freq: {sample['ctrl_freq'].item()} Hz")
        else:
            print("Loader initialized but is empty. Check your file stems and paths.")
    except Exception as e:
        print(f"Critical Loader Error: {e}")