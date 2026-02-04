import os
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

# --- CONFIGURATION ---
DATA_DIR = "/home/e12434694/fp3_pipeline/data"
# Path to your T5 weights (e.g., "google/t5-v1_1-xxl" or a local path)
T5_MODEL_PATH = "google/t5-v1_1-xxl" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard ManiSkill Evaluation Prompts
TASK2LANG = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

def get_mapped_instruction(raw_instr):
    raw_instr = raw_instr.lower()
    if any(k in raw_instr for k in ["insert", "peg", "hole"]):
        return TASK2LANG["PegInsertionSide-v1"]
    if any(k in raw_instr for k in ["stack", "on top", "place on"]):
        return TASK2LANG["StackCube-v1"]
    if any(k in raw_instr for k in ["plug", "socket", "charger", "slot", "kit"]):
        return TASK2LANG["PlugCharger-v1"]
    if any(k in raw_instr for k in ["push", "move", "slide"]):
        return TASK2LANG["PushCube-v1"]
    if any(k in raw_instr for k in ["pick", "grasp", "lift", "cube", "block"]):
        return TASK2LANG["PickCube-v1"]
    return None

def encode_and_save():
    # 1. Load T5 Model
    print(f"Loading T5-XXL from {T5_MODEL_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
    model = T5EncoderModel.from_pretrained(T5_MODEL_PATH).to(DEVICE).eval()

    data_path = Path(DATA_DIR)
    h5_files = list(data_path.rglob('episode_data.h5'))
    
    print(f"Encoding instructions for {len(h5_files)} episodes...")

    # Cache for embeddings to avoid re-encoding the same 5 strings 3,200 times
    embedding_cache = {}

    for h5_path in tqdm(h5_files):
        # Determine output path (lang_embed.npy in the same folder as the h5)
        output_path = h5_path.parent / "lang_embed.npy"
        
        # Skip if already exists
        if output_path.exists():
            continue

        try:
            with h5py.File(h5_path, 'r') as f:
                raw_instr = f.attrs.get('language_instruction', "perform task")
                if isinstance(raw_instr, bytes):
                    raw_instr = raw_instr.decode('utf-8')
            
            mapped_instr = get_mapped_instruction(raw_instr)
            
            if mapped_instr is None:
                continue

            # Check cache
            if mapped_instr not in embedding_cache:
                tokens = tokenizer(
                    mapped_instr, return_tensors="pt", 
                    padding="max_length", max_length=1024, truncation=True
                ).input_ids.to(DEVICE)
                
                with torch.no_grad():
                    embeddings = model(tokens).last_hidden_state.cpu().numpy()
                embedding_cache[mapped_instr] = embeddings

            # Save the (1, 1024, 4096) embedding
            np.save(output_path, embedding_cache[mapped_instr])
            
        except Exception as e:
            print(f"Error processing {h5_path}: {e}")

if __name__ == "__main__":
    encode_and_save()