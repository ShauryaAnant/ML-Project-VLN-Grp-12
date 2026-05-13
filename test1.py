import habitat
import os
from omegaconf import OmegaConf

# Force WSL GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_baseline():
    # Load config
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")

    # ✅ UNFREEZE CONFIG (THIS IS THE KEY FIX)
    OmegaConf.set_readonly(config, False)

    # ✅ Modify config
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1

    # (optional) freeze again
    OmegaConf.set_readonly(config, True)

    with habitat.Env(config=config) as env:
        print("Environment created successfully!")
        observations = env.reset()
        print("Agent is acting inside the environment using random actions...")

        step_count = 0
        while not env.episode_over:
            action = env.action_space.sample()
            observations = env.step(action)
            step_count += 1

        metrics = env.get_metrics()
        print(f"\nEpisode finished after {step_count} steps.")
        print("--- Evaluation Metrics ---")
        print(f"Success Rate (SR): {metrics.get('success', 0.0)}")
        print(f"Success weighted by Path Length (SPL): {metrics.get('spl', 0.0)}")

if __name__ == "__main__":
    run_baseline()
