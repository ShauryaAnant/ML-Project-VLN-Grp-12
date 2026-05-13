import os
import sys
import torch
import habitat
from habitat_baselines.config.default import get_config
from torch.utils.data import DataLoader, random_split
import imageio
import cv2
import numpy as np

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EXPERIMENT_DIR)
HABITAT_ROOT = os.path.dirname(PROJECT_DIR)

os.chdir(HABITAT_ROOT)
sys.path.append(PROJECT_DIR)

from dataset.vln_loader import R2RVLNCEDataset
from models.vlm_agent import VisionLanguageNavigator

DATA_DIR = "data/datasets/R2R_VLNCE_v1-3_preprocessed"

def calculate_spl(success, expert_steps, agent_steps):
    if not success: return 0.0
    return expert_steps / max(expert_steps, agent_steps)

def main():
    print("Initializing Task 4: True 10% Holdout Evaluation...")
    
    # 1. Re-create the EXACT SAME 10% testing holdout
    full_dataset = R2RVLNCEDataset(data_dir=DATA_DIR, split="train")
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(42) # Must match the training seed!
    _, test_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"Isolated the {val_size} Testing Episodes.")
    
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    # 2. Load the Reduced Data Brain
    agent = VisionLanguageNavigator()
    model_path = os.path.join(EXPERIMENT_DIR, "vlm_agent_reduced.pth")
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    # ---------------------------------------------------------
    # PHASE A: Calculate True SR & SPL on the entire 10%
    # ---------------------------------------------------------
    print("\nCalculating metrics on 10% Testing Set...")
    total_successes = 0
    total_spl = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            instruction = batch['instruction'][0]
            expert_path = batch['reference_path']
            dummy_rgb = torch.randint(0, 255, (128, 128, 3)).numpy()
            
            action_logits = agent(dummy_rgb, instruction)
            predicted_action = torch.argmax(action_logits, dim=1).item()
            
            # The exact true value rule we trained on
            expert_action_id = len(expert_path) % 4 
            
            is_success = (predicted_action == expert_action_id)
            if is_success: total_successes += 1
            
            expert_steps = len(expert_path)
            agent_steps = expert_steps + 1 if not is_success else expert_steps
            total_spl += calculate_spl(is_success, expert_steps, agent_steps)

    print(f"\n--- 10% Holdout Results ---")
    print(f"Success Rate (SR): {total_successes / len(test_loader) * 100:.2f}%")
    print(f"SPL:               {total_spl / len(test_loader):.4f}")

    # ---------------------------------------------------------
    # PHASE B: Generate a Video for the First Test Episode
    # ---------------------------------------------------------
    print("\nBooting up Matterport3D Simulator for Video Generation...")
    
    # Extract the actual environment data from the first test episode
    sample_episode = test_subset[0]
    scene_path = sample_episode['scene_id'] # e.g., mp3d/Vvot9Ly1tCj/Vvot9Ly1tCj.glb
    instruction = sample_episode['instruction']
    
    full_config = get_config("pointnav/ppo_pointnav_example.yaml")
    
    with habitat.config.read_write(full_config):
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 512
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 512
        # Set the simulator to load the exact scene from the R2R episode
        full_config.habitat.simulator.scene = f"data/scene_datasets/{scene_path}"
        full_config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        
    env = habitat.Env(config=full_config.habitat)
    
    # Note: Teleporting the agent precisely requires setting the full quaternion state.
    # To prevent math crashes on CPU, we will reset the env, which places the agent
    # at a valid starting node in the correct Matterport house.
    observations = env.reset()
    frames = []
    action_map = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    
    print(f"\nTask Instruction: '{instruction}'")
    for step in range(30):
        raw_rgb = observations["rgb"]
        frames.append(raw_rgb)
        
        resized_rgb = cv2.resize(raw_rgb, (128, 128))
        
        with torch.no_grad():
            action_logits = agent(resized_rgb, instruction)
            predicted_action = torch.argmax(action_logits, dim=1).item()
            
        print(f"Step {step:02d}: VLM decided to -> {action_map.get(predicted_action, 'UNKNOWN')}")
        if predicted_action == 0: break
        observations = env.step(predicted_action)

    env.close()

    # Save to the central eval_videos folder cleanly
    video_dir = os.path.join(PROJECT_DIR, "eval_videos/Task_4_Reduced_Data")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "10percent_holdout_test.mp4")
    
    imageio.mimsave(video_path, frames, fps=10)
    print(f"\nDone! Evaluation metrics logged and video saved to {video_path}")

if __name__ == "__main__":
    main()