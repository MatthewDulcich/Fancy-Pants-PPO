import numpy as np
import pyautogui
import time
import mss
import cv2
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import traceback
import json

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
        self.prev_observation = None  # Initialize prev_observation

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
            0: ('left', 0.1),         # Brief press: Left
            1: ('right', 0.1),        # Brief press: Right
            2: ('s', 0.1),            # Brief press: Jump
            3: ('down', 0.1),         # Brief press: Duck
            4: ('left', 1.5),         # Hold: Left
            5: ('right', 1.5),        # Hold: Right
            6: ('s', 1.5),            # Hold: Jump
            7: ('down', 1.5),         # Hold: Duck
            8: (None, 0),             # No-op
        }

        key, duration = action_map[action]
        print(f"Performing action: {action}, Key: {key}, Duration: {duration}")

        # Perform the action
        if key:
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)

        # Capture observation after action
        new_observation = self.get_observation()

        # Ensure prev_observation is initialized
        if self.prev_observation is None:
            self.prev_observation = new_observation

        # Calculate frame difference
        frame_diff = np.mean(np.abs(self.prev_observation - new_observation))
        print(f"Frame difference after action {action}: {frame_diff}")

        # Update previous observation
        self.prev_observation = new_observation

        # Determine reward
        reward = 10 if frame_diff > 5 else -1  # Adjust threshold (e.g., >5)
        if self.get_done():
            reward = 100
            done = True
        else:
            done = False

        return new_observation, reward, done, {}

    # Reset the environment
    def reset(self):
        """
        Reset the environment and return the initial observation.
        """
        print("Resetting the environment...")

        # Simulate pressing a reset key (e.g., restart game)
        pyautogui.press('r')  # Replace with the appropriate reset command
        time.sleep(2)  # Allow time for the reset

        # Capture the initial observation
        self.prev_observation = self.get_observation()  # Initialize prev_observation
        return self.prev_observation
    
    # Visualize the game (get observation)
    def render(self):
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
            # Downscale (smaller resolution for faster computation)
            downscaled_frame = cv2.resize(gray_frame, (config['down_scaled']['width'], config['down_scaled']['height']))
            # Add channel dimension for compatibility
            observation = np.expand_dims(downscaled_frame, axis=0)
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