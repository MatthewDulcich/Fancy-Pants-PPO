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
import config_handler

# Load configuration
config = config_handler.load_config("game_config.json")

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
        action_map = {
            0: 'left',         # press: Left
            1: 'right',        # press: Right
            2: 's',            # press: Jump
            3: 'down',         # press: Duck
            4: 'no_action'     # No-op
        }

        key = action_map[action]
        # print(f"Performing action: {action}, Key: {key}")

        # Perform the action
        if key != 4:
            self.key_toggle(key)

        # Capture observation after action using `get_observation`
        new_observation, original_scale_frame = self.get_observation()

        # Ensure prev_observation is initialized
        if self.prev_observation is None:
            self.prev_observation = new_observation

        # Detect swirlies
        num_swirlies, current_swirlies, collected_swirlies = track_swirlies(original_scale_frame, self.template, self.prev_swirlies)
        prev_swirlies = current_swirlies
        # Detect swirlies (use the same captured observation)
        num_swirlies, current_swirlies, collected_swirlies = track_swirlies(new_observation[0], self.template, self.prev_swirlies)
        self.prev_swirlies = current_swirlies  # Update prev_swirlies

        # Calculate frame difference
        frame_diff = np.mean(np.abs(self.prev_observation - new_observation))
        print(f"Frame difference after action {action}: {frame_diff:.2f}")

        # Update previous observation
        self.prev_observation = new_observation

        episode_reward = 0
        # Determine reward
        if frame_diff > 5:
            episode_reward += 1
        else:
            episode_reward = -1
        if self.get_done():
            print(f"Reward received for completing the level: {episode_reward}")
            episode_reward = 100
            done = True
        else:
            done = False

        # Reward for collecting swirlies
        swirlie_reward = 10 * collected_swirlies
        episode_reward += swirlie_reward
        # print("Swirlies collected:", collected_swirlies)
        # print(f"Swirlie reward: {swirlie_reward}")

        # Store relevant info in info dict
        info = {
            "action": action,
            "num_swirlies": num_swirlies,
            "collected_swirlies": collected_swirlies,
            "swirlie_reward": swirlie_reward,
            "frame_diff": frame_diff,
            "done": done,
            "episode reward": episode_reward,
            "action": action
        }

        return new_observation, episode_reward, done, info

    def reset(self):
        """
        Reset the environment by restarting the Ruffle server and Safari process.
        """
        try:

            # Reset total reward and rewards list
            self.total_reward = 0
            self.rewards_list = []
            
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
    
    # Visualize the game (get observation)
    def render(self):
        pass
    
    # Not used in this implementation
    # Close the observation (closes render)
    def close(self):
        pass
    
    # Get the game window
    def get_observation(self):
        monitor = {
            "top": self.game_location['top'],
            "left": self.game_location['left'],
            "width": self.game_location['width'],
            "height": self.game_location['height']
        }

        # Capture the game region using persistent mss context
        screenshot = self.sct.grab(monitor)

        # Convert to numpy array and keep only the grayscale channel
        frame = np.array(screenshot, dtype=np.uint8)[:, :, :3]  # Use only the first three channels (BGR)

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale the grayscale frame
        downscaled_frame = cv2.resize(
            frame, 
            (config['down_scaled']['width'], config['down_scaled']['height']), 
            interpolation=cv2.INTER_NEAREST
        )

        # Add channel dimension for compatibility
        observation = np.expand_dims(downscaled_frame, axis=0)

        return observation, frame
        
    def get_done(self):
        """
        Check if the screen is black (end of level).
        """
        # Directly capture observation
        observation, _ = self.get_observation()

        # Use numpy operations to calculate average intensity
        avg_intensity = observation.mean()  # More efficient than np.mean(observation)

        # Optimize threshold comparison
        is_black_screen = avg_intensity < 10  # Fine-tune threshold as needed
        return is_black_screen

    def cleanup_resources(self, server_process, safari_process):
        """
        Clean up resources by terminating server and Safari processes.
        """
        try:
            print("Cleaning up resources...")
            
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