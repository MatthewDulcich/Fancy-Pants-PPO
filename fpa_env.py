# Library imports
import numpy as np
import pyautogui
import mss
import cv2
import traceback
import logging
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from collections import deque

# Script imports
from track_swirlies import track_swirlies
import game_env_setup
import enter_game
import launch_fpa_game
import config_handler as config_handler

# Load configuration
config = config_handler.load_config("game_config.json")

class FPAGame(Env):
    def __init__(self, game_location, server_process=None, safari_process=None):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1, 400, 550), dtype=np.uint8)
        self.action_space = Discrete(6)  # Number of actions
        self.key_states = {}  # Initialize empty key states to keep track of key presses
        self.game_location = game_location  # Set game bounds
        self.prev_observation = None  # Initialize prev_observation
        self.total_reward = 0  # Initialize total reward
        self.rewards_list = deque(maxlen=10)  # Initialize rewards list
        self.prev_swirlies = []  # Initialize prev_swirlies
        self.template = cv2.imread("fpa_swirly_template.png", cv2.IMREAD_GRAYSCALE)  # Load the swirly template

        # Add the correct template in grayscale
        self.door_template = cv2.imread("fpa_enter_game_template.png", cv2.IMREAD_GRAYSCALE)

        # Store recent full-res grayscale observations
        self.recent_full_res_observations = deque(maxlen=5)  # Store the last 5 observations

        self.server_process = server_process  # Add server process
        self.safari_process = safari_process  # Add Safari process
        self.sct = mss.mss()  # Create a persistent mss context for faster screen grabs

    # Helper function to toggle key presses
    def key_toggle(self, key):
        if key not in self.key_states or not self.key_states[key]:
            pyautogui.keyDown(key)
            self.key_states[key] = True
        else:
            pyautogui.keyUp(key)
            self.key_states[key] = False

    def step(self, action):
        # action map
        action_map = {
            0: 'left',         # press: Left
            1: 'right',        # press: Right
            2: 's',            # press: Jump
            3: 'down',         # press: Duck
            4: 'up',           # press: Up
            5: 'no_action'     # No-op
        }

        key = action_map[action]

        # Perform the action
        if key != "no_action":
            self.key_toggle(key)

        # Capture observation after action using `get_observation`
        new_observation, original_scale_frame = self.get_observation()

        # Store the original scale frame in the deque
        self.recent_full_res_observations.append(original_scale_frame)

        # Ensure prev_observation is initialized
        if self.prev_observation is None:
            self.prev_observation = new_observation

        # Detect swirlies
        _, current_swirlies, collected_swirlies = track_swirlies(
            original_scale_frame, self.template, self.prev_swirlies
        )
        self.prev_swirlies = current_swirlies

        # Calculate frame difference
        frame_diff = round(np.mean(np.abs(self.prev_observation - new_observation)))

        # Update previous observation
        self.prev_observation = new_observation

        # Calculate overall reward
        reward = 0

        # Determine reward based on frame difference
        frame_diff_threshold = 5
        if frame_diff > frame_diff_threshold:
            reward += round((frame_diff - frame_diff_threshold) * 0.5)
        else:
            reward -= 1

        # Reward for completing the level
        if self.get_done():
            if self.entered_wrong_door():
                reward -= 1000  # Penalize for entering the wrong door
            else:
                reward += 1000  # Reward for completing the level
            done = True
        else:
            done = False

        # Reward for collecting swirlies
        swirlie_reward = 10 * collected_swirlies
        reward += swirlie_reward

        # Update total reward and rewards list
        self.total_reward += reward
        self.rewards_list.append(reward)

        # Store relevant info in info dict
        info = {
            "action": action,
            "swirlies detected": len(current_swirlies),
            "swirlies collected": collected_swirlies,
            "swirlies reward": swirlie_reward,
            "frame difference": frame_diff,
            "done": done,
            "episode reward": reward,
            "total reward": self.total_reward,
            "last 10 rewards": list(self.rewards_list)[-10:]
        }

        return new_observation, reward, done, info

    def entered_wrong_door(self):
        """
        Check if the agent entered the wrong door by comparing recent observations
        with the door template.
        """
        if not self.door_template:
            logging.error("Door template is missing. Cannot check for wrong door entry.")
            return False

        for observation in self.recent_full_res_observations:
            # Match template using OpenCV
            result = cv2.matchTemplate(observation, self.door_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            # Define a similarity threshold (adjust as needed)
            similarity_threshold = 0.8
            if max_val >= similarity_threshold:
                logging.info(f"Wrong door detected with similarity {max_val:.2f}.")
                return True

        return False
    
    def reset(self):
        """
        Reset the environment by restarting the Ruffle server and Safari process.
        """
        try:

            # Reset total reward and rewards list
            self.total_reward = 0
            self.rewards_list = deque(maxlen=10)
            
            # Stop existing processes safely
            self.cleanup_resources(self.server_process, self.safari_process)

            # Restart Ruffle server and Safari browser
            self.server_process, self.safari_process = self._restart_processes()

            # Navigate to the game level
            safari_window = enter_game.get_most_recent_window_by_owner("Safari")
            if not safari_window:
                raise RuntimeError("Failed to detect Safari window during reset.")

            enter_game.enter_game(safari_window, pre_loaded=True)

            # Reinitialize observation
            self.prev_observation, _ = self.get_observation()
            return self.prev_observation

        except Exception as e:
            traceback.print_exc()
            raise

    def _restart_processes(self):
        """
        Helper method to restart the Ruffle server and Safari process.
        """
        try:
            # Restart Ruffle server
            server_process = launch_fpa_game.start_ruffle_host(config['PORT'])

            # Restart Safari browser
            safari_process = game_env_setup.launch_safari_host(config['GAME_URL'])

            return server_process, safari_process

        except Exception as e:
            traceback.print_exc()
            raise
    
    # Get the game window
    def get_observation(self): 
        # TODO: Fix bug between the monitor and screenshot grab, depending on size of screen the screenshot is doubled, my
        # 1080p screen is giving the correct size, while our 4k laptops are giving double the size
        # Potential fix: take a second swirly screenshot for 4k screens and use that as the template
        monitor = {
            "top": self.game_location['top'],
            "left": self.game_location['left'],
            "width": self.game_location['width'],
            "height": self.game_location['height']
        }
        # print("Monitor:", monitor)

        # Capture the game region using persistent mss context
        screenshot = self.sct.grab(monitor)
        # print("Screenshot shape:", screenshot.size)

        # Convert to numpy array and keep only the grayscale channel
        rgb_frame = np.array(screenshot, dtype=np.uint8)[:, :, :3]  # Use only the first three channels (BGR)

        # Convert to grayscale
        grayscale_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale the grayscale frame
        downscaled_frame = cv2.resize(
            grayscale_frame, 
            (config['down_scaled']['width'], config['down_scaled']['height']), 
            interpolation=cv2.INTER_NEAREST
        )

        # Add channel dimension for compatibility
        downscaled_frame = np.expand_dims(downscaled_frame, axis=0)

        return downscaled_frame, grayscale_frame
        
    def entered_wrong_door(self):
        pass
    
    def get_done(self):
        """
        Check if the screen is black (end of level).
        """
        # Directly capture observation
        downscaled_obs, grayscale_obs = self.get_observation()

        avg_intensity = downscaled_obs.mean()  # More efficient than np.mean(observation)

        # Optimize threshold comparison
        is_black_screen = avg_intensity < 10  # Fine-tune threshold as needed
        return is_black_screen

    def cleanup_resources(self, server_process, safari_process):
        """
        Clean up resources by terminating server and Safari processes.
        """
        try:
            print("Clean up function called in env reset...")
            
            # Safely terminate the Ruffle server process
            if server_process:
                if server_process.poll() is None:  # Check if process is still running
                    server_process.terminate()
                    server_process.wait()
                    print("Ruffle server process terminated.")
                else:
                    print("Ruffle server process already terminated.")
            
            # Safely terminate the Safari process
            if safari_process:
                if safari_process.poll() is None:  # Check if process is still running
                    safari_process.terminate()
                    safari_process.wait()
                    print("Safari process terminated.")
                else:
                    print("Safari process already terminated.")
            
            print("All processes terminated successfully.")
        
        except Exception as e:
            print("An error occurred during cleanup:", e)
            traceback.print_exc()