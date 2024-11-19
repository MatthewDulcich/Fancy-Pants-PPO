import numpy as np
import pyautogui
import time
import mss
import cv2
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import traceback

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