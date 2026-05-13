import habitat_sim
import cv2
import os

# Force WSL GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------
# 🏠 CHOOSE YOUR HOUSE HERE
# Look at your folder list and paste any of the names here!
HOUSE_ID = "17DRP5sb8fy" 
# ---------------------------------------------------------

# Notice the double 'mp3d/mp3d' to match your exact Windows folder structure
scene_path = f"data/scene_datasets/mp3d/{HOUSE_ID}/{HOUSE_ID}.glb"

print(f"Loading House: {scene_path}...")

from habitat_sim.utils.settings import default_sim_settings, make_cfg

sim_settings = default_sim_settings.copy()
sim_settings["scene"] = scene_path
sim_settings["default_agent"] = 0
sim_settings["width"] = 512
sim_settings["height"] = 512
sim_settings["sensor_height"] = 1.5
sim_settings["color_sensor"] = True

cfg = make_cfg(sim_settings)

# 🔥 THE WSL2 CPU RENDERING FIX
cfg.sim_cfg.gpu_device_id = -1

try:
    sim = habitat_sim.Simulator(cfg)
except Exception as e:
    print(f"\n❌ CRASH: Could not find the file! Double check this path exists:\n{scene_path}")
    exit()

agent = sim.initialize_agent(0)
navmesh = sim.pathfinder

# Teleport to a safe spot on the floor
point = navmesh.get_random_navigable_point()
state = agent.get_state()
state.position = point
agent.set_state(state)

print(f"\n✅ SUCCESS! Welcome to {HOUSE_ID}.")
print("🎮 CONTROLS: Click the video window, then use 'W' (forward), 'A' (left), 'D' (right). Press 'Q' to quit.")

# Interactive Game Loop
while True:
    obs = sim.get_sensor_observations()
    frame = obs["color_sensor"]
    
    # RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imshow(f"Exploring: {HOUSE_ID}", frame_bgr)

    key = cv2.waitKey(100)

    if key == ord('w'):
        sim.step("move_forward")
    elif key == ord('a'):
        sim.step("turn_left")
    elif key == ord('d'):
        sim.step("turn_right")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
sim.close()
