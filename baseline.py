#comment

import habitat
# import os

# # Force WSL to use your primary GPU (RTX 3050) to prevent EGL crashes
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_baseline():
    # Load the default PointNav configuration for the test scenes
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")

    with habitat.Env(config=config) as env:
        print("Environment created successfully!")
        observations = env.reset()
        print("Agent is acting inside the environment using random actions...")

        step_count = 0
        while not env.episode_over:
            # The agent randomly chooses from: move forward, turn left, turn right, stop
            action = env.action_space.sample()
            observations = env.step(action)
            step_count += 1

        # Fetch and print the required evaluation metrics
        metrics = env.get_metrics()
        print(f"\nEpisode finished after {step_count} steps.")
        print("--- Evaluation Metrics ---")
        print(f"Success Rate (SR): {metrics.get('success', 0.0)}")
        print(f"Success weighted by Path Length (SPL): {metrics.get('spl', 0.0)}")

if __name__ == "__main__":
    run_baseline()
