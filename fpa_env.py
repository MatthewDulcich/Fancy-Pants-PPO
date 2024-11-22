# Library imports
import numpy as np
import pyautogui
import mss
import cv2
import traceback
import json
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# Script imports
from track_swirlies import track_swirlies
import game_env_setup
import enter_game
import launch_fpa_game

config_file = "game_config.json"
with open(config_file, 'r') as file:
        config = json.load(file)

class FPAGame(Env):
    def __init__(self, game_location, server_process=None, safari_process=None):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1, 400, 800), dtype=np.uint8)
        self.action_space = Discrete(5)  # Number of actions
        self.key_states = {}  # Initialize empty key states to keep track of key presses
        self.game_location = game_location  # Set game bounds
        self.prev_observation = None  # Initialize prev_observation
        self.prev_swirlies = []  # Initialize prev_swirlies
        self.template = cv2.imread("swirly.png")
        self.server_process = server_process  # Add server process
        self.safari_process = safari_process  # Add Safari process


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
            0: 'left',         # press: Left
            1: 'right',        # press: Right
            2: 's',            # press: Jump
            3: 'down',         # press: Duck
            4: 'no_action'     # No-op
        }

        key = action_map[action]
        print(f"Performing action: {action}, Key: {key}")
        
        # Perform the action
        if key != 4:
            self.key_toggle(key)
        
        # Capture observation after action
        new_observation = self.get_observation()

        # Ensure prev_observation is initialized
        if self.prev_observation is None:
            self.prev_observation = new_observation

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

        # Detect swirlies
        num_swirlies, current_swirlies, collected_swirlies = track_swirlies(frame, self.template, self.prev_swirlies)
        prev_swirlies = current_swirlies

        # Calculate frame difference
        frame_diff = np.mean(np.abs(self.prev_observation - new_observation))
        print(f"Frame difference after action {action}: {frame_diff:.2f}")

        # Update previous observation
        self.prev_observation = new_observation

        # Determine reward
        reward = 1 if frame_diff > 5 else -1  # Adjust threshold (e.g., >5)
        if self.get_done():
            print(f"Reward received for completing the level: {reward}")
            reward = 100
            done = True
        else:
            done = False

        siwrlie_reward = reward + 10 * collected_swirlies  # Reward for collecting swirlies
        print("Swirlies collected:", collected_swirlies)
        print(f"Swirlie reward: {siwrlie_reward}")

        # store relevant info in info dict
        info = {
            "num_swirlies": num_swirlies,
            "collected_swirlies": collected_swirlies,
            "swirlie_reward": siwrlie_reward,
            "frame_diff": frame_diff,
            "done": done,
            "reward": reward,
            "action": action,
            "reward_sum": reward
        }

        return new_observation, reward, done, info

    def reset(self):
        """
        Reset the environment by restarting the Ruffle server and Safari process.
        """
        print("Resetting the game by restarting Ruffle and Safari...")
        # Stop existing processes
        self.cleanup_resources(self.server_process, self.safari_process)
        
        # Restart Ruffle host and Safari
        self.server_process = launch_fpa_game.start_ruffle_host(config['PORT'])
        self.safari_process = game_env_setup.launch_safari_host(config['GAME_URL'])
        
        # Re-enter the game
        safari_window = enter_game.get_most_recent_window_by_owner("Safari")
        enter_game.enter_game(safari_window)
        
        # Reinitialize observation
        self.prev_observation = self.get_observation()
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