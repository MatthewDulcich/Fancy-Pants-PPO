import logging
import traceback
import cv2
import random
import time
import json
from torch import save
import os

# Import helper functions from other scripts
from fpa_env import FPAGame
import launch_fpa_game
import game_env_setup
import enter_game
import safari_operations
from ppo_model import PPOAgent, collect_rollouts, update_policy
import torch.optim as optim
import config_handler

# Load configuration
config = config_handler.load_config("game_config.json")

# Ensure the logs directory exists
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'training.log'),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w',  # Overwrite logs for each run
    force=True  # Force configuration even if logging is already set
)
# Add this explicitly after setting up logging:
logging.getLogger().handlers[0].flush()

# Define the PPO policy and value networks
def main():
    """
    Main function to set up the environment, enter the tutorial level, and run training
    with timeout-based resets.
    """
    server_process = None
    safari_process = None
    env = None  # Ensure `env` is defined for cleanup in case of an error

    # Ensure the saved models directory exists
    models_dir = os.path.join(os.getcwd(), 'Model Checkpoints')
    os.makedirs(models_dir, exist_ok=True)

    try:
        # Initialize environment and PPO
        logging.info("Starting PPO Training")
        logging.info(f"Hyperparameters: Learning Rate = {3e-4}, Gamma = {0.99}, Epsilon = {0.2}, Rollout Steps = {2048}")
        launch_fpa_game.kill_port(config['PORT'])
        server_process = game_env_setup.start_ruffle_host(config['PORT'])
        safari_process = game_env_setup.launch_safari_host(config['GAME_URL'])
        
        safari_window = enter_game.get_most_recent_window_by_owner("Safari")
        if not safari_window:
            raise RuntimeError("No Safari window found. Exiting.")
        enter_game.enter_game(safari_window, pre_loaded=True)

        canvas_info = {'top': 0, 'left': 0, 'width': 550, 'height': 400}
        if not canvas_info:
            raise ValueError("Failed to fetch canvas info. Exiting.")

        game_location = {
            'top': int(canvas_info['top']),
            'left': int(canvas_info['left']),
            'width': int(canvas_info['width']),
            'height': int(canvas_info['height']),
        }

        # Adjust game location
        safari_coords = safari_operations.get_safari_window_coordinates()
        if not safari_coords:
            raise RuntimeError("Failed to fetch Safari window coordinates. Exiting.")
        adjusted_game_location = {
            'top': game_location['top'] + safari_coords['top'] + 60,
            'left': game_location['left'] + safari_coords['left'],
            'width': game_location['width'],
            'height': game_location['height'],
        }

        # Initialize FPAGame environment
        logging.info("Initializing FPAGame environment...")
        env = FPAGame(adjusted_game_location, safari_process=safari_process, server_process=server_process)

        # Initialize PPO policy and optimizer
        input_dim = config['down_scaled']['width'] * config['down_scaled']['height']
        policy = PPOAgent(input_dim=input_dim, output_dim=env.action_space.n)
        optimizer = optim.Adam(policy.parameters(), lr=3e-4)

        # Training loop
        logging.info("Starting training with timeout-based reset...")
        timeout = config['timeout_mins'] * 60
        start_time = time.time()
        episode_count = 1

        # Initialize action tracker
        action_counts = {action: 0 for action in range(env.action_space.n)}
        
        # Training loop
        while True:
            episode_start_time = time.time()  # Start time for episode
            print(f"Starting episode {episode_count}")

            # Collect rollouts
            states, actions, rewards, log_probs, values, dones = collect_rollouts(env, policy, n_steps=256)
            logging.info(f"Collected rollouts: {len(states)} steps")
            print(f"Collected rollouts: {len(states)} steps")

            # Update action counts
            for action in actions:
                action_counts[action] += 1

            # Log action distribution
            logging.info(f"Action Distribution: {action_counts}")

            # Update policy
            ppo_loss = update_policy(policy, optimizer, states, actions, rewards, log_probs, values, dones)
            logging.info(f"PPO Loss: {ppo_loss:.4f}")

            # Log reward statistics
            max_reward = max(rewards)
            min_reward = min(rewards)
            cumulative_reward = sum(rewards)
            avg_reward = cumulative_reward / len(rewards)

            logging.info(
            f"Episode {episode_count} | Avg Reward: {avg_reward:.2f} | Max Reward: {max_reward:.2f} | " 
            f"Min Reward: {min_reward:.2f} | Cumulative Reward: {cumulative_reward:.2f}")

            # Log last 10 rewards
            last_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
            logging.info(f"Last 10 Rewards: {last_rewards}")

            # Calculate and log episode duration
            episode_duration = time.time() - episode_start_time
            logging.info(f"Episode {episode_count} Duration: {episode_duration:.2f} seconds")

            # Save policy periodically
            if episode_count % 10 == 0:
                save_path = os.path.join(models_dir, f"ppo_policy_episode_{episode_count}.pt")
                save(policy.state_dict(), save_path)
                logging.info(f"Model saved at Episode {episode_count} to {save_path}")

            # Reset if timeout is reached
            if time.time() - start_time > timeout and not dones[-1]:
                logging.info("Timeout reached. Resetting environment...")
                # env.reset()
                start_time = time.time()

            episode_count += 1
            print(f"Finished episode {episode_count}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()

    finally:
        if env:
            env.cleanup_resources(server_process, safari_process)
        elif server_process and safari_process:
            game_env_setup.cleanup(server_process, safari_process)
        logging.info("All processes terminated successfully. Exiting.")

    return env, server_process, safari_process

if __name__ == "__main__":
    env = None
    server_process = None
    safari_process = None
    try:
        env, server_process, safari_process = main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Cleaning up resources...")
        logging.info("Execution interrupted by user. Cleaning up resources...")
    finally:
        if env:
            env.cleanup_resources(server_process, safari_process)
        elif server_process and safari_process:
            game_env_setup.cleanup(server_process, safari_process)
        logging.info("All processes terminated successfully. Exiting.")