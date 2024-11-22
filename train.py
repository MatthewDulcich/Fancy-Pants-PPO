import logging
import traceback
import cv2
import random
import time
import json

# Import helper functions from other scripts
from fpa_env import FPAGame
import launch_fpa_game
import game_env_setup
import enter_game
import safari_operations

# Load configuration
config_file = "game_config.json"
with open(config_file, 'r') as file:
    config = json.load(file)

logging.basicConfig(
    filename="game_training.log",
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
                logging.info(f"Timeout reached for Episode {episode_count}. Resetting environment...")
                obs = env.reset()
                logging.info(f"Episode {episode_count} Summary: Total Reward = {reward_sum}, Steps = {len(episode_rewards)}")
                episode_count += 1
                reward_sum = 0
                episode_rewards = []
                start_time = time.time()  # Restart timer
                continue

            # Perform a random action
            action = random.randint(0, env.action_space.n - 1)
            obs, reward, done, info = env.step(action)

            # Update rewards
            reward_sum += reward
            episode_rewards.append(reward)

            # Log each action
            logging.info(
                f"Action = {action}, Reward = {reward}, Total Reward = {reward_sum}, Done = {done}"
            )

            # Save observation (optional, for debugging)
            # cv2.imwrite(f"step_{int(time.time() - start_time)}.png", obs[0])

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