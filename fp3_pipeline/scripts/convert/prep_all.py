import os
import h5py
import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
import pickle

# --- CONFIG ---
DATA_DIR = "/home/e12434694/fp3_pipeline/data"
T5_PATH = "/home/e12434694/RoboticsDiffusionTransformer/google/t5-v1_1-xxl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    path = Path(DATA_DIR)
    h5_files = list(path.rglob('episode_data.h5'))
    all_qpos = []
    
    print(f"Loading local T5 from {T5_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
    
    # A40 supports bfloat16 perfectly. Better than float16 for T5.
    model = T5EncoderModel.from_pretrained(
        T5_PATH, 
        torch_dtype=torch.bfloat16, 
        local_files_only=True
    ).to(DEVICE).eval()
    
    cache = {}

    print(f"Processing {len(h5_files)} episodes...")
    for h5_path in tqdm(h5_files):
        try:
            embed_path = h5_path.parent / "lang_embed.npy"
            
            with h5py.File(h5_path, 'r') as f:
                # Key alignment with your previous worker
                state_data = f['state'][:]
                all_qpos.append(state_data[:, 0:7])
                
                raw_instr = f.attrs.get('language_instruction', "perform task")
                if isinstance(raw_instr, bytes): raw_instr = raw_instr.decode('utf-8')
            
            # Save encoding time by checking if we already did this one
            if not embed_path.exists():
                mapped = get_instr(raw_instr)
                if mapped not in cache:
                    # Move tokens to same device as model
                    tok = tokenizer(mapped, return_tensors="pt", padding="max_length", 
                                    max_length=512, truncation=True).input_ids.to(DEVICE)
                    with torch.no_grad():
                        out = model(tok).last_hidden_state
                        cache[mapped] = out.detach().cpu().numpy().astype(np.float32)
                np.save(embed_path, cache[mapped])
                
        except Exception as e:
            print(f"Error in {h5_path}: {e}")

    if all_qpos:
        all_qpos = np.concatenate(all_qpos, axis=0)
        stats = {
            "qpos": {
                "min": np.min(all_qpos, axis=0), 
                "max": np.max(all_qpos, axis=0), 
                "mean": np.mean(all_qpos, axis=0), 
                "std": np.std(all_qpos, axis=0)
            }
        }
        with open("/home/e12434694/fp3_pipeline/stats.pkl", "wb") as f:
            pickle.dump(stats, f)
        print("\nPre-processing finished. stats.pkl and embeddings generated.")

if __name__ == "__main__":
    run_prep()