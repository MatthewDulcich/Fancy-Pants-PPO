# Library imports
import numpy as np
import pyautogui
import mss
import cv2
import traceback
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from collections import deque
import os

# Script imports
from track_swirlies import track_swirlies
import game_env_setup
import enter_game
import launch_fpa_game
import config_handler as config_handler
from logging_config import configure_logging

# Load configuration
config = config_handler.load_config("game_config.json")

# Configure logging
logging, _ = configure_logging()  # Ignore the log filename if not needed here

# Example usage in fpa_env.py
logging.info("FPA environment initialized.")

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
        self.template = cv2.imread("image_templates/swirly.png")  # Load the swirly template

        # Add the correct template in grayscale
        self.door_template = cv2.imread("fpa_enter_game_template.png", cv2.IMREAD_GRAYSCALE)

        # Store recent full-res grayscale observations
        self.recent_full_res_observations = deque(maxlen=5)  # Store the last 5 observations

        self.server_process = server_process  # Add server process
        self.safari_process = safari_process  # Add Safari process
        self.sct = mss.mss()  # Create a persistent mss context for faster screen grabs
        self.repeat_action_window = 10  # Window size for checking repeated actions
        self.recent_actions = deque(maxlen=10)  # Track recent actions
        # self.i = 0  # Initialize counter for debugging

        # Load checkpoint images
        self.checkpoints = []
        self.checkpoint_rewards = set()  # Track rewarded checkpoints
        checkpoint_dir = "checkpoints"
        for i in range(1, 13):
            checkpoint_path = os.path.join(checkpoint_dir, f"Checkpoint{i}.png")
            checkpoint_image = cv2.imread(checkpoint_path)
            if checkpoint_image is not None:
                checkpoint_image = cv2.cvtColor(checkpoint_image, cv2.COLOR_BGR2GRAY)
                self.checkpoints.append(checkpoint_image)

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
        if key != 'no_action':
            self.key_toggle(key)

        # Capture observation after action using `get_observation`
        new_observation, original_scale_frame = self.get_observation()

        # save the original scale frame .png
        # cv2.imwrite(f"original_scale_frame_{self.i}.png", original_scale_frame)
        # self.i += 1

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

        # REWARD LOGIC
        
        # Initialize reward
        reward = 0

        # Reward for hitting the right key
        if action == 1:  # 'right' action
            reward += 2  # Slightly higher reward to encourage progression

        # Frame difference reward
        frame_diff_threshold = 5
        if frame_diff > frame_diff_threshold:
            reward += (frame_diff - frame_diff_threshold) * 0.5  # Scaled reward
        else:
            reward -= (frame_diff_threshold - frame_diff) * 0.2  # Gradual penalty

        # Reward for completing the level
        if self.check_for_black_screen():
            if self.entered_wrong_door():
                reward -= 500  # Reduced penalty for exploration
                done = True
            else:
                reward += 500  # Normalized reward for completion
                done = True
        else:
            done = False

        # Swirlie collection reward
        swirlie_reward =  10 * collected_swirlies  # Scaled reward
        reward += swirlie_reward

        # Reward for reaching a checkpoint
        checkpoint_reward, checkpoint_id = self.checkpoint_matching(original_scale_frame)
        reward += checkpoint_reward
        # print(f"Checkpoint reward: {checkpoint_reward}")

        # Penalty for repeated actions
        if len(self.recent_actions) == self.repeat_action_window and all(a == action for a in self.recent_actions):
            reward -= 2  # Reduced penalty to prevent harsh discouragement

        # Update rewards
        self.total_reward += reward
        self.rewards_list.append(reward)

        # Update recent actions
        self.recent_actions.append(action)

        # Update total reward and rewards list
        self.total_reward += reward
        self.rewards_list.append(reward)

        # Store relevant info in info dict
        info = {
            "action": action,
            "swirlies detected": len(current_swirlies),
            "swirlies collected": collected_swirlies,
            "swirlies reward": swirlie_reward,
            "checkpoint id": checkpoint_id,
            "checkpoint reward": checkpoint_reward,
            "frame difference": frame_diff,
            "done": done,
            "episode reward": reward,
            "total reward": self.total_reward,
            "last 10 rewards": list(self.rewards_list)[-10:],
            "cumulative reward": sum(self.rewards_list)
        }

        return new_observation, reward, done, info

    def entered_wrong_door(self):
        """
        Check if the agent entered the wrong door by comparing recent observations
        with the door template.
        """
        for observation in self.recent_full_res_observations:
            # Match template using OpenCV
            result = cv2.matchTemplate(observation, self.door_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            # Define a similarity threshold (adjust as needed)
            similarity_threshold = 0.9
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
            self.recent_actions.clear()  # Clear recent actions

            # Reset checkpoint list
            # Load checkpoint images
            self.checkpoints = []
            self.checkpoint_rewards = set()  # Track rewarded checkpoints
            checkpoint_dir = "checkpoints"
            for i in range(1, 13):
                checkpoint_path = os.path.join(checkpoint_dir, f"Checkpoint{i}.png")
                checkpoint_image = cv2.imread(checkpoint_path)
                if checkpoint_image is not None:
                    checkpoint_image = cv2.cvtColor(checkpoint_image, cv2.COLOR_BGR2GRAY)
                    self.checkpoints.append(checkpoint_image)
            
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
    
    def check_for_black_screen(self):
        """
        Check if the screen is black (entered a door, game over, change level, etc.)
        """
        # Directly capture observation
        downscaled_obs, grayscale_obs = self.get_observation()

        avg_intensity = downscaled_obs.mean()  # More efficient than np.mean(observation)

        # Optimize threshold comparison
        is_black_screen = avg_intensity < 20  # Fine-tune threshold as needed
        return is_black_screen
    
    def checkpoint_matching(self, observation, print_and_save=False):
        """
        Perform template matching for checkpoints and return the reward if a checkpoint is matched.
        
        Args:
            observation (numpy.ndarray): The current frame of the game.
        
        Returns:
            checkpoint_reward (int): The reward for reaching a checkpoint.
        """
        checkpoint_reward = 0
        threshold = 0.8  # Adjust the threshold as needed
        checkpoint_id = None

        # Ensure observation is a valid NumPy array
        gray_observation = np.array(observation, dtype=np.uint8)

        for idx, checkpoint in enumerate(self.checkpoints):
            # Perform template matching
            result = cv2.matchTemplate(gray_observation, checkpoint, cv2.TM_CCOEFF_NORMED)
            
            # Find all locations where the match exceeds the threshold
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold and idx not in self.checkpoint_rewards:
                self.checkpoint_rewards.add(idx)
                checkpoint_reward += 100  # Adjust the reward value as needed
                checkpoint_id = idx + 1
                if print_and_save:
                    print(f"Checkpoint {idx + 1} reached with score {max_val:.2f}")

                    # Save observations with matching checkpoints drawn
                    annotated_dir = "annotated_images"
                    os.makedirs(annotated_dir, exist_ok=True)

                    # Draw a rectangle around the matched region
                    top_left = max_loc
                    bottom_right = (top_left[0] + checkpoint.shape[1], top_left[1] + checkpoint.shape[0])
                    annotated_image = observation.copy()
                    cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)

                    # Save the annotated image
                    annotated_path = os.path.join(annotated_dir, f"annotated_checkpoint_{idx + 1}.png")
                    cv2.imwrite(annotated_path, annotated_image)

        return checkpoint_reward, checkpoint_id

    def cleanup_resources(self, server_process, safari_process):
        """
        Clean up resources by terminating server and Safari processes.
        """
        def terminate_process(process, name):
            """
            Helper function to safely terminate a process.
            """
            if process:
                if process.poll() is None:  # Check if process is still running
                    try:
                        process.terminate()
                        process.wait(timeout=5)  # Add a timeout to avoid indefinite waiting
                        logging.info(f"{name} process terminated successfully.")
                    except Exception as e:
                        logging.warning(f"Failed to terminate {name} process gracefully: {e}")
                        process.kill()  # Force terminate if graceful termination fails
                        logging.info(f"{name} process killed.")
                else:
                    logging.info(f"{name} process already terminated.")
            else:
                logging.info(f"No {name} process to terminate.")

        try:
            logging.info("Starting resource cleanup...")
            terminate_process(server_process, "Ruffle server")
            terminate_process(safari_process, "Safari")
            logging.info("All processes terminated successfully.")

        except Exception as e:
            logging.error("An error occurred during cleanup:", exc_info=True)