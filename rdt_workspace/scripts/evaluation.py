import sys
import os
from pathlib import Path
import yaml
import torch
import numpy as np
import gymnasium as gym
import tqdm
from collections import deque
from PIL import Image

# --- 1. PATH SETUP ---
REPO_ROOT = "/home/e12434694/RoboticsDiffusionTransformer"
WORKSPACE_ROOT = "/home/e12434694/rdt_workspace"

if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)
if WORKSPACE_ROOT not in sys.path: sys.path.insert(0, WORKSPACE_ROOT)

from scripts.maniskill_model import create_model
import mani_skill.envs 

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-n", "--num-traj", type=int, default=25)
    parser.add_argument("-b", "--sim-backend", type=str, default="gpu")
    parser.add_argument("--pretrained_path", type=str, required=True)
    return parser.parse_args()

args = parse_args()

# --- 2. ENVIRONMENT & MODEL SETUP ---
env = gym.make(
    args.env_id, 
    obs_mode="rgb", 
    control_mode="pd_joint_pos", 
    sim_backend=args.sim_backend
)

config_path = Path(REPO_ROOT) / "configs" / "base.yaml"
if not config_path.exists():
    config_path = Path(REPO_ROOT) / "configs" / "base.yaml"

with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)

# CRITICAL: Match the checkpoint's 3-camera training (4374 tokens)
config['common']['img_cond_len'] = 4374 

policy = create_model(
    args=config, 
    dtype=torch.bfloat16,
    pretrained=args.pretrained_path,
    pretrained_text_encoder_name_or_path=os.path.join(REPO_ROOT, "google/t5-v1_1-xxl"),
    pretrained_vision_encoder_name_or_path=os.path.join(REPO_ROOT, "google/siglip-so400m-patch14-384")
)

task2lang = {"PickCube-v1": "Grasp a red cube and move it to a target goal position."}
text_embed = policy.encode_instruction(task2lang.get(args.env_id, "perform task"))

# --- 3. EVALUATION LOOP ---
success_count = 0
print(f"Starting 3-Camera Evaluation on {args.env_id}...")

for episode in tqdm.trange(args.num_traj):
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = episode + 20241201)
    policy.reset()
    done = False

    while not done:
        sensors = obs['sensor_data']
        
        # A. Extract specific RGB views from your 6-camera base
        rgb_base = np.squeeze(sensors['base_camera']['rgb'].cpu().numpy())
        wrist_key = 'hand_camera' if 'hand_camera' in sensors else 'base_camera'
        rgb_wrist = np.squeeze(sensors[wrist_key]['rgb'].cpu().numpy())

        # B. Map to the 3 slots the model expects
        # Slot 1: Static High | Slot 2: Static High (redundant) | Slot 3: Wrist
        curr_frame_set = [rgb_base, rgb_base, rgb_wrist]
        
        if len(obs_window) == 0: 
            obs_window.append(curr_frame_set)
        obs_window.append(curr_frame_set)

        # C. Build sequence (3 views x 2 timesteps = 6 total images)
        images = []
        for i in range(3):
            images.append(Image.fromarray(obs_window[0][i]).resize((384, 384)))
            images.append(Image.fromarray(obs_window[1][i]).resize((384, 384)))
        
        # D. State Preparation (1D Tensor for internal model unsqueezing)

        raw_qpos = obs['agent']['qpos'].cpu().numpy().flatten()
        
  
        proprio_input = np.zeros(8, dtype=np.float32)
        proprio_input[:min(len(raw_qpos), 8)] = raw_qpos[:min(len(raw_qpos), 8)]
        

        proprio_tensor = torch.from_numpy(proprio_input).to(torch.bfloat16).unsqueeze(0)

        # E. Model Inference
        actions = policy.step(proprio_tensor, images, text_embed).squeeze(0).cpu().numpy()
        
        # F. Step Simulation
        for idx in range(4):
            obs, reward, terminated, truncated, info = env.step(actions[idx])
            if terminated or truncated:
                succ = info.get('success', False)
                if isinstance(succ, np.ndarray): succ = succ.any()
                if succ: success_count += 1
                done = True
                break

print(f"\nFinal Success Rate: {(success_count/args.num_traj)*100:.2f}%")