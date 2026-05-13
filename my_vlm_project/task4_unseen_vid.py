import os
import sys

# ---------------------------------------------------------
# THE FIX: Master Path Override
# ---------------------------------------------------------
# 1. Find exactly where this script lives (.../my_vlm_project)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Find the main habitat folder one level up (.../habitat-lab)
HABITAT_ROOT = os.path.dirname(PROJECT_DIR)
# 3. Force Python to pretend it is running from the habitat root
os.chdir(HABITAT_ROOT)
# 4. Tell Python where your custom models are so imports don't break
sys.path.append(PROJECT_DIR)

import torch
import habitat
from habitat_baselines.config.default import get_config
import imageio
import cv2
import numpy as np

# Now this imports cleanly!
from models.vlm_agent import VisionLanguageNavigator

def generate_evaluation_video():
    print("Initializing Task 4: Unseen Environment Generalization...")
    
    # 1. Load the "Brain" (Point it back to your project folder)
    agent = VisionLanguageNavigator()
    model_path = os.path.join(PROJECT_DIR, "vlm_agent_best.pth")
    try:
        agent.load_state_dict(torch.load(model_path, weights_only=True))
        agent.eval()
        print(f"Successfully loaded trained weights from {model_path}")
    except FileNotFoundError:
        print("ERROR: Could not find vlm_agent_best.pth. Run train.py first!")
        return

    # 2. Load environment configuration
    print("Loading environment configuration...")
    full_config = get_config("pointnav/ppo_pointnav_example.yaml")
    
    with habitat.config.read_write(full_config):
        # High resolution for video
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 512
        full_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 512
        
        # Because we changed the working directory, we DON'T need ../ anymore!
        full_config.habitat.dataset.data_path = "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
        
        # Force CPU Rendering
        full_config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
        
    env = habitat.Env(config=full_config.habitat)
    
    # 3. Task 4 Generalization Test: Custom Instruction
    unseen_instruction = "Walk forward into the living room and turn left."
    print(f"\nTask: {unseen_instruction}")
    
    # 4. The Simulation Loop
    observations = env.reset()
    frames = []
    
    action_map = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    
    print("\nStarting Navigation...")
    for step in range(30):
        raw_rgb = observations["rgb"]
        frames.append(raw_rgb)
        
        resized_rgb = cv2.resize(raw_rgb, (128, 128))
        
        with torch.no_grad():
            action_logits = agent(resized_rgb, unseen_instruction)
            predicted_action = torch.argmax(action_logits, dim=1).item()
            
        print(f"Step {step:02d}: VLM decided to -> {action_map.get(predicted_action, 'UNKNOWN')}")
        
        if predicted_action == 0:
            print("Agent decided to STOP. Episode finished.")
            break
            
        observations = env.step(predicted_action)

    env.close()

    # 5. Save the Video back inside your project folder!
    video_dir = os.path.join(PROJECT_DIR, "eval_videos/Task_4_Generalization")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "unseen_environment_test.mp4")
    
    print(f"\nSaving visualization to {video_path}...")
    imageio.mimsave(video_path, frames, fps=10)
    print("Done! Open the eval_videos folder to see your agent in action.")

if __name__ == "__main__":
    generate_evaluation_video()