
import habitat_sim
import cv2
import numpy as np
import os

# 1. Force WSL GPU visibility for later tasks
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from habitat_sim.utils.settings import default_sim_settings, make_cfg

sim_settings = default_sim_settings.copy()
sim_settings["scene"] = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
sim_settings["default_agent"] = 0
sim_settings["width"] = 512
sim_settings["height"] = 512
sim_settings["sensor_height"] = 1.5
sim_settings["color_sensor"] = True

cfg = make_cfg(sim_settings)

# 🔥 THE WSL2 MAC-TO-WINDOWS FIX 🔥
# Force the raw C++ engine to use CPU rendering and bypass the blocked EGL bridge
cfg.sim_cfg.gpu_device_id = -1

sim = habitat_sim.Simulator(cfg)

agent = sim.initialize_agent(0)
navmesh = sim.pathfinder

# TRY MULTIPLE RANDOM POINTS
for i in range(20):
    point = navmesh.get_random_navigable_point()

    state = agent.get_state()
    state.position = point
    agent.set_state(state)

    obs = sim.get_sensor_observations()
    frame = obs["color_sensor"]

    # Show preview quickly (Converting RGB to BGR for OpenCV)
    # Note: Added color correction here so the apartment isn't blue!
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imshow("Test Spawn", frame_bgr)
    cv2.waitKey(200)

    print("Trying point:", point)

print("Pick a good view and explore!")

# Now normal control
while True:
    obs = sim.get_sensor_observations()
    frame = obs["color_sensor"]
    
    # Convert colors properly for Windows OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imshow("Habitat Live", frame_bgr)

    key = cv2.waitKey(100)

    # Note: In raw habitat_sim, you step the simulator, not the agent!
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