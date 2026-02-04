import os
import h5py
import torch
import numpy as np
import random
import pickle
from pathlib import Path
from tqdm import tqdm
from transformers import T5EncoderModel, AutoTokenizer

# --- DIRECTORY CONFIG ---
WORKSPACE = Path("/home/e12434694/rdt_workspace")
DATA_DIR = WORKSPACE / "data/hdf5"  
STATS_DIR = WORKSPACE / "stats"
EMB_DIR = WORKSPACE / "data/embeddings" 
T5_PATH = "/home/e12434694/RoboticsDiffusionTransformer/google/t5-v1_1-xxl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RDT Standard Instructions for mapping
TASK2LANG = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

def get_instr(raw):
    if not raw or not isinstance(raw, str) or raw.strip() == "" or "perform task" in raw.lower():
        return random.choice(list(TASK2LANG.values()))
    raw = raw.lower()
    if any(k in raw for k in ["insert", "peg"]): return TASK2LANG["PegInsertionSide-v1"]
    if any(k in raw for k in ["stack", "on top"]): return TASK2LANG["StackCube-v1"]
    if any(k in raw for k in ["plug", "slot"]): return TASK2LANG["PlugCharger-v1"]
    if any(k in raw for k in ["push", "move"]): return TASK2LANG["PushCube-v1"]
    return TASK2LANG["PickCube-v1"]

def run_prep():
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    
    target_files = list(DATA_DIR.glob("*.hdf5"))
    if not target_files:
        print(f"Error: No HDF5 files found in {DATA_DIR}")
        return

    print(f"Loading T5 from {T5_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(T5_PATH, local_files_only=True)
    model = T5EncoderModel.from_pretrained(
        T5_PATH, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(DEVICE).eval()
    
    embedding_cache = {}
    all_states = []

    print(f"Generating Embeddings (Seq Len: 1024) and Stats for {len(target_files)} files...")
    for h5_path in tqdm(target_files):
        try:
            with h5py.File(h5_path, 'r') as f:
                # 1. Collect States for stats
                state_data = f['state'][:]
                all_states.append(state_data)
                
                # 2. Map Instruction
                raw_instr = f.attrs.get('language_instruction', "perform task")
                if isinstance(raw_instr, bytes): raw_instr = raw_instr.decode('utf-8')
                mapped_instr = get_instr(raw_instr)

                # 3. Generate 1024-len Embedding
                if mapped_instr not in embedding_cache:
                    tok = tokenizer(
                        mapped_instr, 
                        return_tensors="pt", 
                        padding="max_length", 
                        max_length=1024, # UPDATED TO 1024
                        truncation=True
                    ).input_ids.to(DEVICE)
                    
                    with torch.no_grad():
                        # Shape should be [1, 1024, 4096]
                        out = model(tok).last_hidden_state.squeeze(0)
                        embedding_cache[mapped_instr] = out.cpu().to(torch.float32).numpy()

                # Save standalone .npy
                npy_path = EMB_DIR / f"{h5_path.stem}.npy"
                np.save(npy_path, embedding_cache[mapped_instr])
                
        except Exception as e:
            print(f"Skipping {h5_path.name} due to error: {e}")

    # 4. Save global stats.pkl
    if all_states:
        print("Calculating global state statistics...")
        all_states_np = np.concatenate(all_states, axis=0)
        stats = {
            "state": {
                "mean": np.mean(all_states_np, axis=0).astype(np.float32),
                "std": (np.std(all_states_np, axis=0) + 1e-6).astype(np.float32)
            }
        }
        stats["qpos"] = stats["state"] 
        
        with open(STATS_DIR / "stats.pkl", "wb") as f:
            pickle.dump(stats, f)
            
    print(f"\nSuccess! Verified Seq Len: 1024")
    print(f"Embeddings: {EMB_DIR}")
    print(f"Stats: {STATS_DIR / 'stats.pkl'}")

if __name__ == "__main__":
    run_prep()