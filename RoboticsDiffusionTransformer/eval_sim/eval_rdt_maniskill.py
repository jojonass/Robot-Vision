from typing import Callable, List, Type
import sys
sys.path.append('/')
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import torch
from collections import deque
from PIL import Image
import cv2
import imageio 
import random
import os
import tqdm

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb")
    parser.add_argument("-n", "--num-traj", type=int, default=25)
    parser.add_argument("--only-count-success", action="store_true")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="gpu")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--shader", default="default", type=str)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=0)
    return parser.parse_args()


args = parse_args()
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create video directory
video_dir = "eval_videos"
os.makedirs(video_dir, exist_ok=True)

task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

env_id = args.env_id
env = gym.make(
    env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode=args.render_mode,
    reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    sim_backend='gpu'
)

# Model initialization
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

if os.path.exists(f'text_embed_{env_id}.pt'):
    text_embed = torch.load(f'text_embed_{env_id}.pt')
else:
    text_embed = policy.encode_instruction(task2lang[env_id])
    torch.save(text_embed, f'text_embed_{env_id}.pt')

MAX_EPISODE_STEPS = 400 
total_episodes = args.num_traj  
success_count = 0  
base_seed = 20241201

for episode in tqdm.trange(total_episodes):
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = episode + base_seed)
    policy.reset()

    img = env.render().squeeze(0).detach().cpu().numpy()
    obs_window.append(None)
    obs_window.append(np.array(img))
    proprio = obs['agent']['qpos'][:, :-1]

    global_steps = 0
    video_frames = [] # List to store frames for the current episode
    
    # Store initial frame
    video_frames.append(img)

    done = False
    while global_steps < MAX_EPISODE_STEPS and not done:
        image_arrs = []
        for window_img in obs_window:
            image_arrs.append(window_img)
            image_arrs.append(None)
            image_arrs.append(None)
        
        images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]
        actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
        actions = actions[::4, :]
        
        for idx in range(actions.shape[0]):
            action = actions[idx]
            obs, reward, terminated, truncated, info = env.step(action)
            img = env.render().squeeze(0).detach().cpu().numpy()
            
            obs_window.append(img)
            video_frames.append(img) # Collect frame
            
            proprio = obs['agent']['qpos'][:, :-1]
            global_steps += 1
            
            if terminated or truncated:
                if info.get('success', False):
                    success_count += 1
                done = True
                break 
    
    # Save the video for the episode
    status = "success" if info.get('success', False) else "fail"
    video_path = os.path.join(video_dir, f"trial_{episode+1}_{status}.mp4")
    imageio.mimsave(video_path, video_frames, fps=20) # You can adjust FPS as needed

    print(f"Trial {episode+1} finished, success: {info.get('success', False)}, steps: {global_steps}")

success_rate = success_count / total_episodes * 100
print(f"Success rate: {success_rate}%")
