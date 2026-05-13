import os
import sys
import torch
import habitat
from habitat.core.dataset import Episode
from habitat_baselines.config.default import get_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower # THE ORACLE!
import numpy as np
import json
import tqdm

from dataset.vln_loader import R2RVLNCEDataset
from models.vlm_agent import VisionLanguageNavigator

# 1. SETUP PATHS
EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task4_reduced_data")
HABITAT_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
DATA_DIR = "data/datasets/R2R_VLNCE_v1-3_preprocessed"

os.chdir(HABITAT_ROOT)

class DummyGoal:
    def __init__(self, position):
        self.position = position
        self.radius = 0.5 # A generous half-meter radius to count as "arrived"

def main():
    print("Initializing Offline Geometric Feature Extractor...")
    
    full_dataset = R2RVLNCEDataset(data_dir=DATA_DIR, split="train")
    filtered_episodes = [ep for ep in full_dataset.episodes if "17DRP5sb8fy" in ep['scene_id']]
    print(f"Found {len(filtered_episodes)} total episodes in 17DRP5sb8fy.")

    agent = VisionLanguageNavigator()
    agent.eval()
    
    full_config = get_config("pointnav/ppo_pointnav_example.yaml")
    with habitat.config.read_write(full_config):
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 128
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 128
        full_config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        full_config.habitat.simulator.scene = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
        
    env = habitat.Env(config=full_config.habitat)
    
    # INITIALIZE THE ORACLE
    follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)
    
    features_dir = os.path.join(EXPERIMENT_DIR, "precomputed_features")
    os.makedirs(features_dir, exist_ok=True)
    print(f"Saving features & True Actions to: {features_dir}\n")

    for ep_idx, sample_episode in enumerate(tqdm.tqdm(filtered_episodes)):
        ep_id = sample_episode['episode_id']
        feature_path = os.path.join(features_dir, f"{ep_id}.pt")
        
        if os.path.exists(feature_path): continue

        # NO MORE 2-METER HACK! We find the true final destination of the human path
        final_destination = sample_episode['reference_path'][-1]

        custom_episode = Episode(
            episode_id=ep_id,
            scene_id=f"data/scene_datasets/{sample_episode['scene_id']}",
            start_position=sample_episode['start_position'],
            start_rotation=sample_episode['start_rotation']
        )
        custom_episode.goals = [DummyGoal(position=final_destination)]
        
        env.episodes = [custom_episode]
        observations = env.reset()
        
        episode_features = []
        episode_actions = []
        
        # Follow the true 3D path until we reach the goal
        while True:
            # 1. Take picture & process visual feature
            raw_rgb = observations["rgb"][..., :3].copy()
            with torch.no_grad():
                inputs = agent.processor(images=raw_rgb, return_tensors="pt")
                vis_feat = agent.clip_model.get_image_features(**inputs)
                episode_features.append(vis_feat.squeeze(0))
                
            # 2. Ask the Oracle for the exact geometric action to reach the final_destination
            best_action = follower.get_next_action(final_destination)
            
            # If we reached the goal (or Oracle says STOP), we are done with this episode
            if best_action is None or best_action == 0:
                episode_actions.append(0) # 0 is STOP
                break
                
            episode_actions.append(best_action)
            
            # 3. Take the physical step
            observations = env.step(best_action)
            
            # Failsafe: Don't get stuck in an infinite loop if the NavMesh gets confused
            if len(episode_actions) > 40:
                break
                
        # SAVE BOTH THE IMAGES AND THE TRUE ACTIONS!
        torch.save({
            "features": torch.stack(episode_features),  # Shape: (Steps, 512)
            "actions": torch.tensor(episode_actions)    # Shape: (Steps,)
        }, feature_path)

    env.close()
    print("\nFeature & Action extraction complete! All hacks removed.")

if __name__ == "__main__":
    main()