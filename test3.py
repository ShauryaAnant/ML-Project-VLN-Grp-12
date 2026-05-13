import habitat
import os
from omegaconf import OmegaConf
import imageio  # ✅ NEW: For making the GIF
import random   # ✅ NEW: To control the random actions

# Force WSL GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_baseline():
    # Load config
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")

    # UNFREEZE CONFIG
    OmegaConf.set_readonly(config, False)
    # CPU Render Fallback
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
    # FREEZE CONFIG
    OmegaConf.set_readonly(config, True)

    with habitat.Env(config=config) as env:
        print("\nEnvironment created successfully!")
        observations = env.reset()
        
        frames = []
        # Add the very first frame to our movie
        frames.append(observations["rgb"])
        
        print("🎬 Recording agent actions... please wait a few seconds!")

        step_count = 0
        # Run for exactly 50 steps
        while not env.episode_over and step_count < 50:
            # Force the agent to only pick Forward (1), Left (2), or Right (3). No Stopping!
            action = random.choice([1, 2, 3])
            observations = env.step(action)
            
            # Save the current view to our movie frames
            frames.append(observations["rgb"])
            step_count += 1

        # Stitch all the frames together into an animated GIF!
        imageio.mimsave("agent_exploration.gif", frames, fps=10)
        
        print(f"\n✅ SUCCESS: Episode finished after {step_count} steps.")
        print("🎞️ Video saved as 'agent_exploration.gif'!")

if __name__ == "__main__":
    run_baseline()