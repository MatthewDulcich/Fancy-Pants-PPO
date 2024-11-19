# import mss # Can use mss for screen cap
import pyautogui
import time
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium import Env
import traceback
import cv2
import mss
import matplotlib.pyplot as plt
import random
import json

# Import helper functions from other scripts
import launch_fpa_game
import game_env_setup
import enter_game
import safari_operations

config_file = "game_config.json"
with open(config_file, 'r') as file:
        config = json.load(file)

class FPAGame(Env):
    def __init__(self, game_location):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1, 400, 800), dtype=np.uint8)
        self.action_space = Discrete(7)  # Number of actions
        self.key_states = {}  # Initialize empty key states to keep track of key presses
        self.game_location = game_location  # Set game bounds

    # Helper function to toggle key presses
    def key_toggle(self, key):
        if key not in self.key_states or not self.key_states[key]:
            pyautogui.keyDown(key)
            self.key_states[key] = True
        else:
            pyautogui.keyUp(key)
            self.key_states[key] = False

    def step(self, action):
        action_map = {
            0: ['left'],         # Brief press: Left
            1: ['right'],        # Brief press: Right
            2: ['s'],            # Brief press: Jump
            3: ['down'],         # Brief press: Duck
            4: ['left'],         # Hold: Left
            5: ['right'],        # Hold: Right
            6: ['s'],            # Hold: Jump
            7: ['down'],         # Hold: Duck
            8: [],               # No-op
        }

        # Ensure game window is in focus
        pyautogui.click(x=self.game_location['left'] + 60, y=self.game_location['top'] + 60)

        # Debug: Print action and keys
        print(f"Performing action: {action}, Key(s): {action_map[action]}")

        # Perform the action
        if action in [4, 5, 6, 7]:  # Hold actions
            for key in action_map[action]:
                pyautogui.keyDown(key)
            time.sleep(1.5)  # Adjust hold duration
            for key in action_map[action]:
                pyautogui.keyUp(key)
        elif action in [0, 1, 2, 3]:  # Brief press actions
            for key in action_map[action]:
                pyautogui.keyDown(key)
            time.sleep(0.1)  # Brief press
            for key in action_map[action]:
                pyautogui.keyUp(key)

        # Capture the next observation
        prev_obs = self.get_observation()
        observation = self.get_observation()

        # Debug: Check observation difference
        diff = np.sum(np.abs(prev_obs - observation))
        print(f"Frame difference after action {action}: {diff}")

        # Check if the game is in the finished state
        done = self.get_done()

        # Reward logic
        if done:
            reward = 100  # Large reward for completing the level
        elif diff > 0:
            reward = 10  # Reward for visible progress
        else:
            reward = -1  # Penalize no progress

        info = {}
        return observation, reward, done, info
    
    # Visualize the game (get observation)
    def render(self):
        pass
    # Reset the game
    def reset(self):
        pass
    # Close the observation (closes render)
    def close(self):
        pass
    # Get the game window
    def get_observation(self):
        with mss.mss() as sct:
            monitor = {
                "top": self.game_location['top'],
                "left": self.game_location['left'],
                "width": self.game_location['width'],
                "height": self.game_location['height']
            }
            # Capture the game region
            screenshot = sct.grab(monitor)
            # Convert to numpy array to fetch pixel data
            frame = np.array(screenshot)[:, :, :3]
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # Resize to match observation space
            resized_frame = cv2.resize(gray_frame, (800, 600))  # Width x Height
            # Add channel dimension for compatibility
            observation = np.expand_dims(resized_frame, axis=0)
            return observation
        
    def get_done(self):
        """
        Check if the screen is black (end of level).
        """
        observation = self.get_observation()
        # Calculate the average pixel intensity
        avg_intensity = np.mean(observation)
        # Set a threshold for detecting a black screen
        black_screen_threshold = 10  # Fine-tune this value based on testing
        return avg_intensity < black_screen_threshold

    def cleanup_resources(self, server_process, safari_process):
        """
        Clean up resources by terminating server and Safari processes.
        """
        try:
            print("Cleaning up resources...")
            if server_process:
                server_process.terminate()
                server_process.wait()
            if safari_process:
                safari_process.terminate()
                safari_process.wait()
            print("All processes terminated successfully.")
        except Exception as e:
            print("An error occurred during cleanup:", e)
            traceback.print_exc()

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

        # Step 2: Start Safari WebDriver
        print("Starting Safari WebDriver and navigating to game URL...")
        safari_process, driver = game_env_setup.start_safari_webdriver(config['GAME_URL'])

        # Step 3: Automate entering the tutorial level
        print("Automating game entry to reach the tutorial level...")
        safari_window = enter_game.get_most_recent_window_by_owner("Safari")
        if not safari_window:
            raise Exception("No Safari window found. Exiting...")
        
        enter_game.enter_game(safari_window)  # Navigate to the tutorial level

        # Step 4: Fetch canvas information and content offset
        print("Fetching game canvas size and position...")
        canvas_info = game_env_setup.fetch_canvas_position_and_size(driver)
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
            'top': game_location['top'] + safari_coords['top'],
            'left': game_location['left'] + safari_coords['left'],
            'width': game_location['width'],
            'height': game_location['height'],
        }
        print("Adjusted Game Location:", adjusted_game_location)

        # Step 5: Initialize the FPAGame environment
        print("Initializing FPAGame environment...")
        env = FPAGame(adjusted_game_location)

        # Step 6: Capture initial observation to verify setup
        print("Capturing initial observation from the game...")
        obs = env.get_observation()
        plt.imshow(obs[0], cmap='gray')  # Display the first channel as grayscale
        plt.title("Initial Observation")
        # plt.show()

        # Step 7: Run 10 random actions
        print("Running 10 random actions in the environment...")
        rewards = 0
        for i in range(10):
            action = random.randint(0, env.action_space.n - 1)  # Random action
            obs, reward, done, info = env.step(action)
            rewards += reward

            # Display the observation and action info
            print(f"Step {i+1}: Action = {action}, Reward = {reward}, Cumulative Rewards = {rewards}, Done = {done}")
            plt.imshow(obs[0], cmap='gray')
            plt.title(f"Step {i+1}: Action {action}")
            # plt.show()

            if done:
                print("Finished the level!")
                break

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()

    finally:
        # Cleanup resources
        print("Cleaning up resources...")
        if env:
            env.cleanup_resources(server_process, safari_process)
        elif server_process and safari_process:
            game_env_setup.cleanup(server_process, safari_process)

        print("All processes terminated successfully.")

if __name__ == "__main__":
    main()