import os
import sys
import torch
import habitat
from habitat.core.dataset import Episode
from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from torch.utils.data import DataLoader
import imageio
import cv2
import numpy as np
import json
import textwrap

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EXPERIMENT_DIR)
HABITAT_ROOT = os.path.dirname(PROJECT_DIR)

os.chdir(HABITAT_ROOT)
sys.path.append(PROJECT_DIR)

from dataset.vln_loader import R2RVLNCEDataset
from models.vlm_agent import VisionLanguageNavigator

DATA_DIR = "data/datasets/R2R_VLNCE_v1-3_preprocessed"

class DummyGoal:
    def __init__(self, position):
        self.position = position
        self.radius = 0.5 

def main():
    print("Initializing Task 4: Presentation-Ready Holdout Evaluation...")
    
    test_ids_path = os.path.join(EXPERIMENT_DIR, "test_holdout_ids.json")
    with open(test_ids_path, "r") as f:
        test_ids = json.load(f)
        
    full_dataset = R2RVLNCEDataset(data_dir=DATA_DIR, split="train")
    test_episodes = [ep for ep in full_dataset.episodes if ep['episode_id'] in test_ids]
    full_dataset.episodes = test_episodes
    print(f"Successfully loaded the exact {len(test_episodes)} holdout episodes!")
    
    agent = VisionLanguageNavigator()
    model_path = os.path.join(EXPERIMENT_DIR, "vlm_agent_reduced.pth")
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    print(f"\nBooting up Matterport3D Simulator...")
    
    full_config = get_config("pointnav/ppo_pointnav_example.yaml")
    with habitat.config.read_write(full_config):
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 512
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 512
        full_config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        full_config.habitat.simulator.scene = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
        
    env = habitat.Env(config=full_config.habitat)
    follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)
    
    video_dir = os.path.join(PROJECT_DIR, "eval_videos/Task_4_Reduced_Data")
    os.makedirs(video_dir, exist_ok=True)
    
    # METRIC TRACKERS
    global_correct_actions = 0
    global_total_actions = 0
    
    for ep_idx, sample_episode in enumerate(test_episodes):
        instruction = sample_episode['instruction']['instruction_text']
        print(f"\n=== Generating Video {ep_idx + 1}/{len(test_episodes)} ===")
        print(f"Task: '{instruction}'")
        
        final_destination = sample_episode['reference_path'][-1]
        
        custom_episode = Episode(
            episode_id=f"vlm_test_{ep_idx}",
            scene_id=f"data/scene_datasets/{sample_episode['scene_id']}",
            start_position=sample_episode['start_position'],
            start_rotation=sample_episode['start_rotation']
        )
        custom_episode.goals = [DummyGoal(position=final_destination)]
        
        env.episodes = [custom_episode]
        observations = env.reset()
        
        frames = []
        action_map = {0: "STOP", 1: "FWD", 2: "LEFT", 3: "RIGHT"}
        memory_state = None
        
        for step in range(15): 
            raw_rgb = observations["rgb"][..., :3].copy() 
            resized_rgb = cv2.resize(raw_rgb, (128, 128))
            
            with torch.no_grad():
                action_logits, memory_state = agent(resized_rgb, instruction, hidden_state=memory_state)
                predicted_action = torch.argmax(action_logits, dim=1).item()
                
                # Convert logits to probabilities (0.0 to 1.0)
                probs = torch.softmax(action_logits, dim=1)[0].numpy()
                
            best_action = follower.get_next_action(final_destination)
            if best_action is None: best_action = 0
            
            is_optimal = (predicted_action == best_action)
            global_total_actions += 1
            if is_optimal: global_correct_actions += 1
            
            # Get Distance to Goal from the Oracle
            agent_pos = env.sim.get_agent_state().position
            dist_to_goal = np.linalg.norm(np.array(agent_pos) - np.array(final_destination))
            
            # -----------------------------------------------------
            # DRAWING THE HUD
            # -----------------------------------------------------
            # 1. Create a semi-transparent black overlay for readability
            overlay = raw_rgb.copy()
            cv2.rectangle(overlay, (0, 0), (512, 110), (0, 0, 0), -1)   # Top box
            cv2.rectangle(overlay, (0, 420), (512, 512), (0, 0, 0), -1) # Bottom box
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, raw_rgb, 1 - alpha, 0, raw_rgb)
            
            # 2. Top HUD: Instruction (Wrapped)
            wrapped_text = textwrap.wrap(f"Task: {instruction}", width=60)
            for i, line in enumerate(wrapped_text):
                cv2.putText(raw_rgb, line, (10, 25 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            cv2.putText(raw_rgb, f"Ep: {ep_idx+1}/8 | Step: {step+1}/15", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(raw_rgb, f"Dist to Goal: {dist_to_goal:.2f}m", (300, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 3. Bottom HUD: Probabilities
            prob_str = f"STOP:{probs[0]*100:.0f}% | FWD:{probs[1]*100:.0f}% | LFT:{probs[2]*100:.0f}% | RGT:{probs[3]*100:.0f}%"
            cv2.putText(raw_rgb, prob_str, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 4. Bottom HUD: Decisions
            agent_str = action_map.get(predicted_action, "UNK")
            oracle_str = action_map.get(best_action, "UNK")
            color = (0, 255, 0) if is_optimal else (255, 0, 0) # Green if correct, Red if wrong
            
            cv2.putText(raw_rgb, f"Agent: {agent_str}", (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(raw_rgb, f"Oracle: {oracle_str}", (300, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            frames.append(raw_rgb)
            if predicted_action == 0: break
            observations = env.step(predicted_action)

        video_path = os.path.join(video_dir, f"annotated_holdout_ep{ep_idx+1}.mp4")
        imageio.mimsave(video_path, frames, fps=2)
        print(f"Saved: {video_path}")

    env.close()
    print(f"\nAll {len(test_episodes)} videos generated successfully!")
    
    # FINAL METRICS
    final_accuracy = (global_correct_actions / global_total_actions) * 100
    print("\n========================================")
    print("FINAL HOLDOUT EVALUATION METRICS")
    print("========================================")
    print(f"Total Steps Taken: {global_total_actions}")
    print(f"Optimal Actions Taken: {global_correct_actions}")
    print(f"Action Accuracy: {final_accuracy:.2f}%")
    print("========================================")

if __name__ == "__main__":
    main()