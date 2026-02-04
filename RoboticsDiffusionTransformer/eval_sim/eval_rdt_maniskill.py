from typing import Callable, List, Type
import sys
import os
import random
import argparse
import yaml
import torch
import numpy as np
import gymnasium as gym
import tqdm
import csv
import imageio
from collections import deque
from PIL import Image
from pathlib import Path

# Maniskill and Model imports
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help="Environment to run.")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true", help="Only save successful trajectories.")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Simulation backend.")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="Rendering mode.")
    parser.add_argument("--shader", default="default", type=str, help="Shader used for rendering.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()

# --- Initialization ---
args = parse_args()
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setup directories for results
workspace_root = Path("/home/e12434694/rdt_workspace")
eval_dir = workspace_root / "evaluation_results" / args.env_id
eval_dir.mkdir(parents=True, exist_ok=True)

task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

# --- Environment Setup ---
env_id = args.env_id
env = gym.make(
    env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode=args.render_mode,
    reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    sim_backend="gpu"
)

# --- Model Setup ---
config_path = 'configs/base.yaml'
with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)

policy = create_model(
    args=config, 
    dtype=torch.bfloat16,
    pretrained=args.pretrained_path,
    pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl",
    pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384"
)

# --- Instruction Encoding ---
if os.path.exists(f'text_embed_{env_id}.pt'):
    text_embed = torch.load(f'text_embed_{env_id}.pt')
else:
    text_embed = policy.encode_instruction(task2lang[env_id])
    torch.save(text_embed, f'text_embed_{env_id}.pt')

# --- Evaluation Loop ---
MAX_EPISODE_STEPS = 400 
total_episodes = args.num_traj  
success_count = 0  
base_seed = 20241201

# Flags to ensure we only save one of each video type
saved_success_video = False
saved_failure_video = False
eval_stats = []

print(f"Starting evaluation on {env_id} for {total_episodes} trajectories...")

for episode in tqdm.trange(total_episodes):
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = episode + base_seed)
    policy.reset()

    # Initial frame capture
    img = env.render().squeeze(0).detach().cpu().numpy()
    obs_window.append(None)
    obs_window.append(np.array(img))
    proprio = obs['agent']['qpos'][:, :-1]

    global_steps = 0
    video_frames = [img] # Buffer to store frames for video
    done = False
    is_success = False

    while global_steps < MAX_EPISODE_STEPS and not done:
        image_arrs = []
        for window_img in obs_window:
            image_arrs.append(window_img)
            image_arrs.append(None)
            image_arrs.append(None)
        
        images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]
        
        # Policy Inference
        actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
        
        # Execute action chunks (Subsampling for RDT temporal consistency)
        actions = actions[::4, :]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record frame
            img = env.render().squeeze(0).detach().cpu().numpy()
            video_frames.append(img)
            
            # Update state
            obs_window.append(img)
            proprio = obs['agent']['qpos'][:, :-1]
            global_steps += 1
            
            if terminated or truncated:
                if info.get('success', False):
                    is_success = True
                    success_count += 1
                done = True
                break 

    # --- Video Saving Logic ---
    if is_success and not saved_success_video:
        vid_path = eval_dir / f"success_trial_{episode}.mp4"
        imageio.mimsave(vid_path, video_frames, fps=20)
        saved_success_video = True
        print(f" Saved success video to {vid_path}")
    
    if not is_success and not saved_failure_video:
        vid_path = eval_dir / f"failure_trial_{episode}.mp4"
        imageio.mimsave(vid_path, video_frames, fps=20)
        saved_failure_video = True
        print(f" Saved failure video to {vid_path}")

    # Log statistics
    eval_stats.append({
        "trial": episode + 1,
        "success": is_success,
        "steps": global_steps,
        "seed": episode + base_seed
    })

# --- Final Reporting ---
success_rate = (success_count / total_episodes) * 100

# Save stats to CSV
csv_path = eval_dir / "results.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["trial", "success", "steps", "seed"])
    writer.writeheader()
    writer.writerows(eval_stats)

print("\n" + "="*40)
print(f"FINAL RESULTS FOR {env_id}")
print(f"Success Rate: {success_rate:.2f}% ({success_count}/{total_episodes})")
print(f"Stats saved to: {csv_path}")
print(f"Videos saved in: {eval_dir}")
print("="*40)