import traceback
import cv2
import matplotlib.pyplot as plt
import random
import time
import json

# Import helper functions from other scripts
from fpa_env import FPAGame
import launch_fpa_game
import game_env_setup
import enter_game
import safari_operations
# import mute_safari_tab

config_file = "game_config.json"
with open(config_file, 'r') as file:
        config = json.load(file)

def main():
    """
    Main function to set up the environment, enter the tutorial level, and capture observations
    while performing 10 random actions.
    """
    server_process = None
    safari_process = None
    env = None  # Ensure `env` is defined for cleanup in case of an error

    try:
        # Step 1: Ensure the port is free and start the Ruffle server
        print("Cleaning up the port and starting Ruffle server...")
        launch_fpa_game.kill_port(config['PORT'])  # Ensure the port is available
        server_process = game_env_setup.start_ruffle_host(config['PORT'])

        # Step 2: Launch the game in Safari
        print("Launching the game in Safari...")
        safari_process = game_env_setup.launch_safari_host(config['GAME_URL'])

        # Step 3: Automate entering the tutorial level
        print("Automating game entry to reach the tutorial level...")
        safari_window = enter_game.get_most_recent_window_by_owner("Safari")
        if not safari_window:
            raise Exception("No Safari window found. Exiting...")
        
        # mute_safari_tab.mute_specific_tab(driver)  # Mute the game tab

        enter_game.enter_game(safari_window, pre_loaded=True)  # Navigate to the tutorial level

        # Step 4: Fetch canvas information and content offset
        print("Fetching game canvas size and position...")
        # canvas_info = game_env_setup.fetch_canvas_position_and_size(driver)
        canvas_info = {'top': 0, 'left': 0, 'width': 550, 'height': 400}
        if not canvas_info:
            raise Exception("Failed to fetch canvas info. Exiting...")

        game_location = {
            'top': int(canvas_info['top']),
            'left': int(canvas_info['left']),
            'width': int(canvas_info['width']),
            'height': int(canvas_info['height']),
        }
        print("Game Location (Canvas Info):", game_location)

        # Fetch content offset and adjust the game location
        print("Fetching content offset for browser adjustments...")
        safari_coords = safari_operations.get_safari_window_coordinates()
        if not safari_coords:
            raise Exception("Failed to fetch Safari window coordinates. Exiting...")
        adjusted_game_location = {
            'top': game_location['top'] + safari_coords['top'] + 60,  # Adjust for menu bar
            'left': game_location['left'] + safari_coords['left'],
            'width': game_location['width'],
            'height': game_location['height'],
        }
        print("Adjusted Game Location:", adjusted_game_location)

        # Step 5: Initialize the FPAGame environment
        print("Initializing FPAGame environment...")
        env = FPAGame(adjusted_game_location, safari_process=safari_process, server_process=server_process)

        # Step 6: Capture initial observation to verify setup
        print("Capturing initial observation from the game...")
        obs = env.get_observation()

        # Step 7: Run random actions with a timeout
        print("Running actions with timeout-based reset...")
        timeout = config['timeout_mins'] * 60  # 2 minutes timeout for each episode
        start_time = time.time()  # Track start time of the episode
        reward_sum = 0  # Track the total reward for the episode
        rewards_tracker = []  # Track rewards for plotting

        # create a log txt file
        log_file = open("log.txt", "w")

        while True:
            # Check for timeout
            if time.time() - start_time > timeout:
                print("Timeout reached! Resetting the environment...")
                obs = env.reset()  # Reset the environment
                start_time = time.time()  # Restart the timer
                continue  # Start the next episode

            # Perform a random action
            action = random.randint(0, 4)
            obs, reward, done, info = env.step(action)

            # Save observation locally
            cv2.imwrite(f"step_{int(time.time() - start_time)}.png", obs[0])

            # Track rewards for plotting
            rewards_tracker.append(reward_sum)
            reward_sum += reward
            log_file.write(f"Action = {action}, Reward = {reward}, Done = {done}\n")
            log_file.write(f"Total Reward: {reward_sum}\n")
            log_file.write(50 * "-")
            log_file.flush()  # Ensure the log is written to the file
            
            # End the episode if the level is finished
            if done:
                print("Finished the level! Resetting the environment...")
                obs = env.reset()
                start_time = time.time()  # Restart the timer

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()

    finally:
        if env:
            env.cleanup_resources(server_process, safari_process)
        elif server_process and safari_process:
            game_env_setup.cleanup(server_process, safari_process)

if __name__ == "__main__":
    main()