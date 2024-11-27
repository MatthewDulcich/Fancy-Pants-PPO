import logging
import traceback
import cv2
import random
import time
import json
import os

# Import helper functions from other scripts
from fpa_env import FPAGame
import launch_fpa_game
import game_env_setup
import enter_game
import safari_operations
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

def main():
    """
    Main function to set up the environment, enter the tutorial level, and run training
    with timeout-based resets.
    """
    server_process = None
    safari_process = None
    env = None  # Ensure `env` is defined for cleanup in case of an error

    try:
        # Step 1: Ensure the port is free and start the Ruffle server
        logging.info("Starting Ruffle server...")
        launch_fpa_game.kill_port(config['PORT'])
        server_process = game_env_setup.start_ruffle_host(config['PORT'])

        # Step 2: Launch the game in Safari
        logging.info("Launching the game in Safari...")
        safari_process = game_env_setup.launch_safari_host(config['GAME_URL'])

        # Step 3: Automate entering the tutorial level
        logging.info("Automating game entry...")
        safari_window = enter_game.get_most_recent_window_by_owner("Safari")
        if not safari_window:
            raise RuntimeError("No Safari window found. Exiting.")
        enter_game.enter_game(safari_window, pre_loaded=True)

        # Step 4: Fetch canvas information and content offset
        logging.info("Fetching game canvas position and size...")
        canvas_info = {'top': 0, 'left': 0, 'width': 550, 'height': 400}
        if not canvas_info:
            raise ValueError("Failed to fetch canvas info. Exiting.")

        game_location = {
            'top': int(canvas_info['top']),
            'left': int(canvas_info['left']),
            'width': int(canvas_info['width']),
            'height': int(canvas_info['height']),
        }
        logging.info(f"Game Location: {game_location}")

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
        logging.info(f"Adjusted Game Location: {adjusted_game_location}")

        # Step 5: Initialize FPAGame environment
        logging.info("Initializing FPAGame environment...")
        env = FPAGame(adjusted_game_location, safari_process=safari_process, server_process=server_process)

        # Step 6: Run actions with timeout-based resets
        logging.info("Starting training with timeout-based reset...")
        timeout = config['timeout_mins'] * 60
        start_time = time.time()
        reward_sum = 0
        episode_rewards = []  # Track rewards for the current episode
        episode_count = 1

        while True:
            # Check for timeout
            if time.time() - start_time > timeout:
                print(100 * "-")
                positive_rewards = sum(1 for r in episode_rewards if r > 0)
                negative_rewards = sum(1 for r in episode_rewards if r < 0)
                average_reward = reward_sum / len(episode_rewards) if episode_rewards else 0

                logging.info(
                    f"Episode {episode_count} Summary: "
                    f"Total Reward = {reward_sum}, "
                    f"Steps = {len(episode_rewards)}, "
                    f"Positive Rewards = {positive_rewards}, "
                    f"Negative Rewards = {negative_rewards}, "
                    f"Average Reward = {average_reward:.2f}, "
                    f"Max Reward = {max(episode_rewards, default=0)}, "
                    f"Last 10 Rewards = {episode_rewards[-10:]}"
                )
                logging.info(f"Timeout or episode complete for Episode {episode_count}. Resetting environment...")
                logging.info(100 * "-")
                logging.info(100 * "-")
                logging.info(100 * "-")
                obs, reward_sum, episode_rewards, episode_count, start_time = game_env_setup.reset_episode(env, reward_sum, episode_rewards, episode_count)
                continue

            # Perform a random action
            action = random.randint(0, env.action_space.n - 1)
            obs, reward, done, info = env.step(action)

            # Log additional info
            # Step-level logging
            logging.info(
                f"Step {len(episode_rewards)} | "
                f"Action: {info['action']} | "
                f"Reward: {info['episode reward']} | "
                f"Total Episode Reward: {reward_sum} | "
                f"Frame Difference: {info['frame difference']} | "
                f"Swirlies Detected: {info.get('swirlies detected', 0)} | "
                f"Swirlies Collected: {info.get('swirlies collected', 0)} | "
                f"Last 10 Rewards: {episode_rewards[-10:]}"
            )

            # Update rewards
            reward_sum += reward
            episode_rewards.append(reward)

            rolling_window = 10
            rolling_avg_reward = sum(episode_rewards[-rolling_window:]) / len(episode_rewards[-rolling_window:])
            rolling_avg_frame_diff = sum(info.get('frame difference', 0) for _ in episode_rewards[-rolling_window:]) / rolling_window

            logging.info(
                f"Rolling Avg Reward (Last {rolling_window} Steps): {rolling_avg_reward:.2f} | "
                f"Rolling Avg Frame Diff (Last {rolling_window} Steps): {rolling_avg_frame_diff:.2f}"
            )
            
            # End the episode if the level is finished
            if done:
                logging.info(f"Level finished in Episode {episode_count}. Resetting environment...")
                logging.info(f"Episode {episode_count} Summary: Total Reward = {reward_sum}, Steps = {len(episode_rewards)}")
                obs = env.reset()
                episode_count += 1
                reward_sum = 0
                episode_rewards = []
                start_time = time.time()  # Restart timer

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()

    finally:
        # Cleanup resources
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