# import os
# from omegaconf import OmegaConf
# from PIL import Image

# # 1. THE ULTIMATE OVERRIDE: Force Linux to use Software (CPU) Rendering at the OS level
# os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

# import habitat

# def run_baseline():
#     config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    
#     # UNFREEZE CONFIG
#     OmegaConf.set_readonly(config, False)
    
#     # 2. Force the Simulator to render via CPU
#     config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
    
#     # FREEZE CONFIG
#     OmegaConf.set_readonly(config, True)
    
#     print("\n--- Booting Simulator Engine (CPU Mode) ---")
#     try:
#         with habitat.Env(config=config) as env:
#             print("--- 🚀 SUCCESS: Environment Engine is Online! ---")
#             observations = env.reset()
            
#             # Save the starting view
#             img = Image.fromarray(observations["rgb"], 'RGB')
#             img.save("agent_step_0.png")
#             print("\n📸 Saved starting view to 'agent_step_0.png'")
            
#             step_count = 0
#             # Let's run it for exactly 5 steps so you get a sequence of images
#             while not env.episode_over and step_count < 5:
#                 action = env.action_space.sample()
#                 observations = env.step(action)
#                 step_count += 1
                
#                 # Save the new view after taking a step
#                 img2 = Image.fromarray(observations["rgb"], 'RGB')
#                 img2.save(f"agent_step_{step_count}.png")
#                 print(f"📸 Saved view after action to 'agent_step_{step_count}.png'")
                
#             metrics = env.get_metrics()
#             print(f"\nEpisode paused after {step_count} steps.")
#             print("--- Evaluation Metrics ---")
#             print(f"Success Rate (SR): {metrics.get('success', 0.0)}")
#             print(f"SPL: {metrics.get('spl', 0.0)}\n")
#     except Exception as e:
#         print(f"\n❌ ERROR: {e}")

# if __name__ == "__main__":
#     run_baseline()


import habitat
import os
from omegaconf import OmegaConf
from PIL import Image  # ✅ NEW: Library to save images

# Force WSL GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_baseline():
    # Load config
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")

    # UNFREEZE CONFIG
    OmegaConf.set_readonly(config, False)

    # Modify config to bypass the WSL EGL issue
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1

    # freeze again
    OmegaConf.set_readonly(config, True)

    with habitat.Env(config=config) as env:
        print("\nEnvironment created successfully!")
        observations = env.reset()
        
        # 📸 NEW: Take a picture of the starting point
        img_start = Image.fromarray(observations["rgb"], 'RGB')
        img_start.save("agent_start_view.png")
        print("-> Saved starting 3D view to 'agent_start_view.png'")

        print("Agent is acting inside the environment using random actions...")

        step_count = 0
        # ✅ Added a limit of 5 steps so you get a few images to look at
        while not env.episode_over and step_count < 5:
            action = env.action_space.sample()
            observations = env.step(action)
            step_count += 1
            
            # 📸 NEW: Take a picture after every single step
            img_step = Image.fromarray(observations["rgb"], 'RGB')
            img_step.save(f"agent_step_{step_count}.png")
            print(f"-> Saved step {step_count} view to 'agent_step_{step_count}.png'")

        metrics = env.get_metrics()
        print(f"\nEpisode finished after {step_count} steps.")
        print("--- Evaluation Metrics ---")
        print(f"Success Rate (SR): {metrics.get('success', 0.0)}")
        print(f"Success weighted by Path Length (SPL): {metrics.get('spl', 0.0)}\n")

if __name__ == "__main__":
    run_baseline()