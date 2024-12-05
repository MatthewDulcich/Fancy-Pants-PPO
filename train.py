import logging
import traceback
import time
from torch import save
import numpy as np
import os
import torch

# Import helper functions from other scripts
from fpa_env import FPAGame
import launch_fpa_game
import game_env_setup
import enter_game
import safari_operations
from ppo_model import PPO
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

def save_metrics(metrics, save_dir, episode_count):
    """
    Save metrics to a file for later analysis.
    """
    metrics_file = os.path.join(save_dir, "metrics.csv")
    with open(metrics_file, "a") as f:
        if episode_count == 1:  # Write header on the first save
            f.write("Episode,Reward,Length,PolicyLoss,ValueLoss,Entropy\n")
        for i in range(len(metrics["episode_rewards"])):
            f.write(f"{episode_count},{metrics['episode_rewards'][i]},{metrics['episode_lengths'][i]},"
                    f"{metrics['policy_losses'][i]},{metrics['value_losses'][i]},{metrics['entropy'][i]}\n")

def main():
    """
    Main function to set up the environment, enter the tutorial level, and run training
    using the updated PPO structure.
    """
    server_process = None
    safari_process = None
    env = None

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
        
        # Initialize PPO with the new structure
        input_channels = 1  # Assuming grayscale image
        input_height = config['down_scaled']['height']
        input_width = config['down_scaled']['width']
        n_actions = env.action_space.n

        # Create PPO instance
        ppo = PPO(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            n_actions=n_actions,
            lr=config.get('lr', 3e-4),
            gamma=config.get('gamma', 0.99),
            epsilon=config.get('epsilon', 0.2),
            entropy_coef=config.get('entropy_coef', 0.01)
        )

        # Training loop
        logging.info("Starting training with timeout-based reset...")
        timeout = config['timeout_mins'] * 60
        start_time = time.time()

        # Initialize metrics
        episode_count = 1
        metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy": []
        }

        while True:
            logging.info(100*"=")
            logging.info(100*"=")
            logging.info(f"Starting episode {episode_count}")
            
            # Collect rollouts
            states, actions, rewards, log_probs, values, dones = ppo.collect_rollouts(env, n_steps=config['rollout_steps'])
            
            # Update policy and track loss
            policy_loss, value_loss, entropy = ppo.update_policy(
                states=states,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                values=values,
                dones=dones,
                k_epochs=config.get('k_epochs', 4),
                clip_grad=config.get('clip_grad', 0.5)
            )
            
            # Calculate episode-level metrics
            episode_reward = rewards.sum().item()
            episode_length = len(rewards)
            
            # Update metrics dictionary
            metrics["episode_rewards"].append(episode_reward)
            metrics["episode_lengths"].append(episode_length)
            metrics["policy_losses"].append(policy_loss)
            metrics["value_losses"].append(value_loss)
            metrics["entropy"].append(entropy)
            
            # Logging metrics
            logging.info(f"Episode {episode_count} | Reward: {episode_reward:.2f} | Length: {episode_length}")
            logging.info(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
            
            # Save metrics periodically
            if episode_count % config.get('save_interval', 30) == 0:
                save_metrics(metrics, models_dir, episode_count)
                logging.info(f"Metrics saved for Episode {episode_count}")
            
            # Timeout handling
            if time.time() - start_time > timeout:
                logging.info("Timeout reached. Resetting environment...")
                start_time = time.time()
            
            episode_count += 1

    except Exception as e:
        logging.exception(f"An error occurred during training: {e}")
        traceback.print_exc()

    finally:
        if env:
            env.cleanup_resources(server_process, safari_process)
        elif server_process and safari_process:
            game_env_setup.cleanup(server_process, safari_process)
        logging.info("All processes terminated successfully. Exiting.")

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