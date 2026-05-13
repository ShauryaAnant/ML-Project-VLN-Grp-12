import habitat_sim
import cv2
import numpy as np

from habitat_sim.utils.settings import default_sim_settings, make_cfg

sim_settings = default_sim_settings.copy()
sim_settings["scene"] = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
sim_settings["default_agent"] = 0
sim_settings["width"] = 512
sim_settings["height"] = 512
sim_settings["sensor_height"] = 1.5
sim_settings["color_sensor"] = True

cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

agent = sim.initialize_agent(0)
navmesh = sim.pathfinder

# 🔥 TRY MULTIPLE RANDOM POINTS
for i in range(20):
    point = navmesh.get_random_navigable_point()

    state = agent.get_state()
    state.position = point
    agent.set_state(state)

    obs = sim.get_sensor_observations()
    frame = obs["color_sensor"]

    # Show preview quickly
    cv2.imshow("Test Spawn", frame)
    cv2.waitKey(200)

    print("Trying point:", point)

print("Pick a good view and explore!")

# Now normal control
while True:
    obs = sim.get_sensor_observations()
    frame = obs["color_sensor"]
    cv2.imshow("Habitat Live", frame)

    key = cv2.waitKey(100)

    if key == ord('w'):
        agent.act("move_forward")
    elif key == ord('a'):
        agent.act("turn_left")
    elif key == ord('d'):
        agent.act("turn_right")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
sim.close()
