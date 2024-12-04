import logging
import traceback
import time
from torch import save
import numpy as np
import os

# Import helper functions from other scripts
from fpa_env import FPAGame
import launch_fpa_game
import game_env_setup
import enter_game
import safari_operations
from ppo_model import PPOAgent, collect_rollouts, update_policy
import torch.optim as optim
import config_handler as config_handler

# Load configuration
config = config_handler.load_config("game_config.json")

# Ensure the logs directory exists
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

current_time = time.strftime("%Y%m%d-%H%M%S")
log_filename = os.path.join(logs_dir, f"fpa_game_logs_{current_time}.log")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Logs written to a file with a unique name
        logging.StreamHandler()  # Logs displayed in the console
    ],
    level=logging.INFO  # Set default logging level to INFO
)

# Set the logging level for the console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show WARNING or higher in the console
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Get the root logger and replace its handlers
logger = logging.getLogger()
logger.handlers.clear()  # Remove existing handlers
logger.addHandler(console_handler)  # Add the console handler
logger.addHandler(logging.FileHandler(log_filename))  # Add the file handler with a unique name

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
        # # Initialize environment and PPO
        # logging.info("Starting PPO Training")
        # logging.info(f"Hyperparameters: Learning Rate = {3e-4}, Gamma = {0.99}, Epsilon = {0.2}, Rollout Steps = {2048}")
        # launch_fpa_game.kill_port(config['PORT'])
        # server_process = game_env_setup.start_ruffle_host(config['PORT'])
        # safari_process = game_env_setup.launch_safari_host(config['GAME_URL'])
        
        # safari_window = enter_game.get_most_recent_window_by_owner("Safari")
        # if not safari_window:
        #     raise RuntimeError("No Safari window found. Exiting.")
        # enter_game.enter_game(safari_window, pre_loaded=True)

        canvas_info = {'top': 0, 'left': 0, 'width': 550, 'height': 400}
        if not canvas_info:
            logging.error("Failed to fetch canvas info. Exiting.")
            raise ValueError("Failed to fetch canvas info. Exiting.")

        # Adjust game location
        safari_coords = safari_operations.get_safari_window_coordinates()
        if not safari_coords:
            raise RuntimeError("Failed to fetch Safari window coordinates. Exiting.")
        adjusted_game_location = {
            'top': canvas_info['top'] + safari_coords['top'] + 60,
            'left': canvas_info['left'] + safari_coords['left'],
            'width': canvas_info['width'],
            'height': canvas_info['height'],
        }

        # Initialize FPAGame environment
        logging.info("Initializing FPAGame environment...")
        env = FPAGame(adjusted_game_location, safari_process=safari_process, server_process=server_process)
        
        # Initialize PPO policy and optimizer
        input_channels = 1  # Assuming grayscale image, adjust to 3 if RGB
        input_height = config['down_scaled']['height']  # Height of the resized observation
        input_width = config['down_scaled']['width']    # Width of the resized observation
        output_dim = env.action_space.n  # Number of possible actions in the environment

        # Create the PPO policy with convolutional layers
        policy = PPOAgent(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            output_dim=output_dim
        )

        # Optimizer for the policy
        optimizer = optim.Adam(policy.parameters(), lr=1e-3)

        # Training loop
        logging.info("Starting training with timeout-based reset...")
        timeout = config['timeout_mins'] * 60
        start_time = time.time()
        episode_count = 1

        # Initialize action tracker
        action_counts = {action: 0 for action in range(env.action_space.n)}
        
        # Training loop
        while True:
            logging.info(f"Starting episode {episode_count}")

            episode_start_time = time.time()  # Start time for episode
            print(f"Starting episode {episode_count}")

            # Collect rollouts
            states, actions, rewards, log_probs, values, dones = collect_rollouts(env, policy, n_steps=config['rollout_steps'])

            # Normalize rewards
            rewards = np.array(rewards, dtype=np.float32)  # Convert rewards to a NumPy array
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)  # Normalize
            logging.info(f"Normalized rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
            
            if dones[-1]:
                logging.info("Episode completed successfully.")
                cumulative_reward = 0

            # Update action counts
            for action in actions:
                action_counts[action] += 1

            # Log action distribution

            # Update policy
            ppo_loss = update_policy(policy, optimizer, states, actions, rewards, log_probs, values, dones)

            # Log reward statistics
            max_reward = max(rewards)
            min_reward = min(rewards)
            cumulative_reward = sum(rewards)
            avg_reward = cumulative_reward / len(rewards)

            # Log last 10 rewards
            last_rewards = rewards[-10:] if len(rewards) >= 10 else rewards

            # Save policy periodically
            if episode_count % 30 == 0:
                save_path = os.path.join(models_dir, f"ppo_policy_episode_{episode_count}.pt")
                save(policy.state_dict(), save_path)
                logging.info(f"Model saved at Episode {episode_count} to {save_path}")

            # Reset if timeout is reached
            if time.time() - start_time > timeout and not dones[-1]:
                logging.info("Timeout reached. Resetting environment...")
                start_time = time.time()

            episode_count += 1
            print(f"Finished episode {episode_count}")

            logging.info(f"Episode {episode_count} Summary: "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Max Reward: {max_reward:.2f}, "
                f"Min Reward: {min_reward:.2f}, "
                f"Cumulative Reward: {cumulative_reward:.2f}, "
                f"Last 10 Rewards: {last_rewards}")
            logging.info("=" * 100)
            logging.info("=" * 100)

    except Exception as e:
        logging.exception(f"An error occurred during training: {e}")
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