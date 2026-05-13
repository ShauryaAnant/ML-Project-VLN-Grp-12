import habitat_sim
import cv2
import torch
import numpy as np
import os
from torchvision import transforms

# 1. IMPORT YOUR BRAIN
from vln_model import VisionLanguageNavigator

# Force WSL GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("🧠 Waking up the AI Brain...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionLanguageNavigator().to(device)
model.eval() # Put the brain in "testing" mode

# Standard image processing to translate the Simulator Camera into PyTorch format
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("🏠 Booting up Matterport3D Simulator...")
HOUSE_ID = "1LXtFkjw3qL" # The house you used earlier!
scene_path = f"data/scene_datasets/mp3d/mp3d/{HOUSE_ID}/{HOUSE_ID}.glb"

from habitat_sim.utils.settings import default_sim_settings, make_cfg
sim_settings = default_sim_settings.copy()
sim_settings["scene"] = scene_path
sim_settings["width"] = 640
sim_settings["height"] = 480
sim_settings["color_sensor"] = True

cfg = make_cfg(sim_settings)
cfg.sim_cfg.gpu_device_id = -1
sim = habitat_sim.Simulator(cfg)

agent = sim.initialize_agent(0)
navmesh = sim.pathfinder

# Teleport to a safe spot on the floor
point = navmesh.get_random_navigable_point()
state = agent.get_state()
state.position = point
agent.set_state(state)

# Map the Neural Network outputs to Habitat commands
action_map = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right"
}

# The fake instruction we are testing
instruction = ["Walk down the hallway and stop at the stairs"]

print("\n🤖 AI is taking control! Watch the window. Press 'Q' to force quit.")

while True:
    # 1. THE EYES: Get the current camera view
    obs = sim.get_sensor_observations()
    frame_rgba = obs["color_sensor"]
    
    # Show us what the AI sees on screen
    frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
    cv2.imshow("AI Autopilot Vision", frame_bgr)
    
    # 2. PREPARE DATA: Convert image for ResNet18
    frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
    image_tensor = image_transform(frame_rgb).unsqueeze(0).to(device)
    
    # 3. THE BRAIN: Ask PyTorch what to do
    with torch.no_grad():
        logits = model(image_tensor, instruction)
        
    # 4. DECISION: Find the action with the highest score
    chosen_action_idx = torch.argmax(logits, dim=1).item()
    action_name = action_map[chosen_action_idx]
    
    print(f"Action Logits: {np.round(logits.cpu().numpy()[0], 2)} --> 🧠 AI chose: {action_name.upper()}")
    
    # 5. EXECUTE: Tell the simulator to move
    if action_name == "stop":
        print("🛑 AI decided it has reached the goal (or got confused) and STOPPED.")
        break
    else:
        sim.step(action_name)
        
    # Pause for 500ms so the AI doesn't move at the speed of light
    key = cv2.waitKey(500)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
sim.close()
