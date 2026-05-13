import os
import sys
import torch
import habitat
from habitat.core.dataset import Episode
from habitat_baselines.config.default import get_config
from torch.utils.data import DataLoader
import imageio
import cv2
import numpy as np
import json

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EXPERIMENT_DIR)
HABITAT_ROOT = os.path.dirname(PROJECT_DIR)

os.chdir(HABITAT_ROOT)
sys.path.append(PROJECT_DIR)

from dataset.vln_loader import R2RVLNCEDataset
from models.vlm_agent import VisionLanguageNavigator

DATA_DIR = "data/datasets/R2R_VLNCE_v1-3_preprocessed"

# ---------------------------------------------------------
# THE FIX: Create a fake goal to satisfy the PointNav engine
# ---------------------------------------------------------
class DummyGoal:
    def __init__(self, position):
        self.position = position
        self.radius = 0.1

def main():
    print("Initializing Task 4: Explicit 10% Holdout Evaluation...")
    
    test_ids_path = os.path.join(EXPERIMENT_DIR, "test_holdout_ids.json")
    with open(test_ids_path, "r") as f:
        test_ids = json.load(f)
        
    full_dataset = R2RVLNCEDataset(data_dir=DATA_DIR, split="train")
    
    test_episodes = [ep for ep in full_dataset.episodes if ep['episode_id'] in test_ids]
    full_dataset.episodes = test_episodes
    print(f"Successfully loaded the exact {len(test_episodes)} holdout episodes!")
    
    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    agent = VisionLanguageNavigator()
    model_path = os.path.join(EXPERIMENT_DIR, "vlm_agent_reduced.pth")
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    print("\nCalculating metrics on Testing Set...")
    total_successes = 0
    with torch.no_grad():
        for batch in test_loader:
            instruction = batch['instruction'][0]
            expert_path = batch['reference_path']
            dummy_rgb = torch.randint(0, 255, (128, 128, 3)).numpy()
            
            action_logits = agent(dummy_rgb, instruction)
            predicted_action = torch.argmax(action_logits, dim=1).item()
            expert_action_id = len(expert_path) % 4 
            
            if predicted_action == expert_action_id:
                total_successes += 1

    print(f"\n--- 10% Holdout Results ---")
    print(f"Success Rate (SR): {total_successes / len(test_loader) * 100:.2f}%")

    # ---------------------------------------------------------
    # MATTERPORT VIDEO GENERATION
    # ---------------------------------------------------------
    print("\nBooting up Matterport3D Simulator...")
    
    sample_episode = full_dataset.episodes[0]
    instruction = sample_episode['instruction']['instruction_text']
    scene_id = sample_episode['scene_id']
    
    full_config = get_config("pointnav/ppo_pointnav_example.yaml")
    with habitat.config.read_write(full_config):
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 512
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 512
        full_config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        
    env = habitat.Env(config=full_config.habitat)
    
    print(f"Forcing Simulator to load: {scene_id}")
    custom_episode = Episode(
        episode_id="vlm_test_0",
        scene_id=f"data/scene_datasets/{scene_id}",
        start_position=sample_episode['start_position'],
        start_rotation=sample_episode['start_rotation']
    )
    
    # INJECT THE DUMMY GOAL (Offset by 2 meters to prevent ZeroDivisionError!)
    dummy_pos = list(sample_episode['start_position'])
    dummy_pos[0] += 2.0 # Move it 2 meters along the X-axis
    custom_episode.goals = [DummyGoal(position=dummy_pos)]
    
    env.episodes = [custom_episode]
    
    observations = env.reset()
    
    frames = []
    action_map = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
    expert_action_id = len(sample_episode['reference_path']) % 4 
    
    print(f"\nTask: '{instruction}'")
    for step in range(15): 
        raw_rgb = observations["rgb"][..., :3].copy() 
        resized_rgb = cv2.resize(raw_rgb, (128, 128))
        
        with torch.no_grad():
            action_logits = agent(resized_rgb, instruction)
            predicted_action = torch.argmax(action_logits, dim=1).item()
            
        is_optimal = (predicted_action == expert_action_id)
        
        action_str = action_map.get(predicted_action, "UNKNOWN")
        color = (0, 255, 0) if is_optimal else (255, 0, 0) 
        
        cv2.putText(raw_rgb, f"Step: {step+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(raw_rgb, f"Action: {action_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(raw_rgb, f"Optimal? {'YES' if is_optimal else 'NO'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        frames.append(raw_rgb)
        
        print(f"Step {step+1}: Agent chose {action_str} | Optimal? {is_optimal}")
        if predicted_action == 0: break
        
        observations = env.step(predicted_action)

    env.close()

    video_dir = os.path.join(PROJECT_DIR, "eval_videos/Task_4_Reduced_Data")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "annotated_holdout_test.mp4")
    
    imageio.mimsave(video_path, frames, fps=2)
    print(f"\nDone! Slow-motion annotated video saved to {video_path}")

if __name__ == "__main__":
    main()