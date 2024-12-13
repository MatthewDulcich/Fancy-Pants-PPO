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
import wandb
from concurrent.futures import ThreadPoolExecutor

# Script imports
from fpa_env_functions.track_swirlies import track_swirlies
import game_env_setup
import enter_game
import launch_fpa_game
import config_handler as config_handler
from logging_config import configure_logging
from safari_operations import get_safari_window_coordinates
from pytesseract_interaction import get_tab_bar_region, handle_reload_bar
from reward_logic import calculate_rewards

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
        # self.prev_swirlies = []  # Initialize prev_swirlies
        # self.swirles_template = cv2.imread("images/image_templates/swirly_gray.png", cv2.IMREAD_GRAYSCALE)
        # action map
        self.action_map = config['action_map']

        # Add the correct template in grayscale
        self.door_template = cv2.imread("images/image_templates/fpa_exit_game_template.png", cv2.IMREAD_GRAYSCALE)

        # Store recent full-res grayscale observations
        self.recent_full_res_observations = deque(maxlen=5)  # Store the last 5 observations

        self.server_process = server_process  # Add server process
        self.safari_process = safari_process  # Add Safari process
        self.sct = mss.mss()  # Create a persistent mss context for faster screen grabs
        self.repeat_action_window = 10  # Window size for checking repeated actions
        self.recent_actions = deque(maxlen=10)  # Track recent actions
        self.i = 0  # Initialize counter for debugging

        # Load checkpoint images
        self.checkpoints = []
        self.checkpoint_rewards = set()  # Track rewarded checkpoints
        checkpoint_dir = "images/game_checkpoint_images/game_checkpoints"
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
        if self.tab_bar_check():
            return self.reset()

        # Perform the action
        key = self.action_map[str(action)]
        self.key_toggle(key)

        # Capture observation after action
        new_obs, original_scale_gray_obs = self.get_observation()

        # # save the original scale frame .png
        # cv2.imwrite(f"original_scale_gray_obs_{self.i}.png", original_scale_gray_obs)
        # self.i += 1

        # Store the original scale frame in the deque
        self.recent_full_res_observations.append(original_scale_gray_obs)

        # Detect swirlies
        collected_swirlies = 0
        # _, current_swirlies, collected_swirlies = track_swirlies(
        #     original_scale_gray_obs, self.swirles_template, self.prev_swirlies
        # )
        # self.prev_swirlies = current_swirlies

        # # Calculate frame difference
        # frame_diff = round(np.mean(np.abs(self.prev_observation - new_obs)))
        # self.prev_observation = new_obs  # Update previous observation

        # Calculate reward using the external function
        reward, done, info = calculate_rewards(
            original_scale_gray_obs,
            self.recent_full_res_observations,
            # collected_swirlies,
            frame_diff,
            self.checkpoint_matching,
            self.check_for_black_screen,
            self.entered_correct_door
        )

        # Update rewards
        self.total_reward += reward
        self.rewards_list.append(reward)

        info.update({
            "action": action,
            # "frame difference": frame_diff,
            "episode reward": reward,
            "total reward": self.total_reward,
            "frame difference": frame_diff
        })

        # Log relevant information with wandb
        wandb.log(info)

        return new_obs, reward, done, info

    def entered_correct_door(self, recent_observations):
        """
        Check if the agent entered the correct door by comparing the template with the recent observations.
        """
        print("Checking for wrong door entry...")
        for observation in recent_observations:
            # Match template using OpenCV
            result = cv2.matchTemplate(observation, self.door_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            # Define a similarity threshold (adjust as needed)
            similarity_threshold = 0.9
            logging.info(f"Correct door detected with similarity {max_val:.2f}.")
            print(f"Correct door detected with similarity {max_val:.2f}.")
            if max_val >= similarity_threshold:
                print("Correct door detected.")
                return True

        return False
    
    def reset(self):
        """
        Reset the environment by restarting the Ruffle server and Safari process.
        """
        print("Resetting environment...")
        try:

            # Reset total reward and rewards list
            self.total_reward = 0
            self.rewards_list = deque(maxlen=10)
            self.recent_actions.clear()  # Clear recent actions

            # Reset checkpoint list
            # Load checkpoint images
            self.checkpoints = []
            self.checkpoint_rewards = set()  # Track rewarded checkpoints
            checkpoint_dir = "images/game_checkpoint_images/game_checkpoints"
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
        # 1080p screen is giving the correct size, while our 4k laptops are giving double the size
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
    
    def check_for_black_screen(self, obs):
        """
        Check if the screen is black (entered a door, game over, change level, etc.)
        """
        avg_intensity = obs.mean()

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
            checkpoint_id (int): The ID of the matched checkpoint.
        """
        checkpoint_reward = 0
        threshold = 0.9  # Adjust the threshold as needed
        checkpoint_id = None

        # Define progressive rewards for checkpoints (1 to 12)
        progressive_rewards = [30, 50, 70, 100, 120, 150, 180, 220, 260, 300, 350, 400]

        # Ensure observation is a valid NumPy array
        gray_observation = np.array(observation, dtype=np.uint8)

        def match_template(idx, checkpoint):
            # Perform template matching
            result = cv2.matchTemplate(gray_observation, checkpoint, cv2.TM_CCOEFF_NORMED)
            
            # Find all locations where the match exceeds the threshold
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold and idx not in self.checkpoint_rewards:
                self.checkpoint_rewards.add(idx)
                reward = progressive_rewards[idx]  # Get the progressive reward for this checkpoint
                print(f"Checkpoint {idx + 1} reached with score {max_val:.2f}, reward {reward}")
                return idx, max_val, max_loc, reward
            return None

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(match_template, idx, checkpoint) for idx, checkpoint in enumerate(self.checkpoints)]
            for future in futures:
                result = future.result()
                if result:
                    idx, max_val, max_loc, reward = result
                    checkpoint_reward += reward  # Add the progressive reward
                    checkpoint_id = idx + 1
                    if print_and_save:
                        print(f"Checkpoint {idx + 1} reached with score {max_val:.2f}, reward {reward}")

                        # Save observations with matching checkpoints drawn
                        annotated_dir = "images/game_checkpoint_images/annotated_checkpoint_images"
                        os.makedirs(annotated_dir, exist_ok=True)

                        # Draw a rectangle around the matched region
                        top_left = max_loc
                        bottom_right = (top_left[0] + self.checkpoints[idx].shape[1], top_left[1] + self.checkpoints[idx].shape[0])
                        annotated_image = observation.copy()
                        cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)

                        # Save the annotated image
                        annotated_path = os.path.join(annotated_dir, f"annotated_checkpoint_{idx + 1}.png")
                        cv2.imwrite(annotated_path, annotated_image)

        return checkpoint_reward, checkpoint_id
    
    def tab_bar_check(self):
            # Add offset which sometimes triggers
            if config.get('tabs_present', False):
                tab_offset = 25
            else:
                tab_offset = 0
            safari_window = get_safari_window_coordinates()

            tab_bar_region = get_tab_bar_region(safari_window, offset=tab_offset)
            if handle_reload_bar(tab_bar_region):
                print("Handled reload bar. Proceeding to click play again")
                # self.reset()
                return True

    def terminate_process(self, process, name):
        """
        Helper function to safely terminate a process.
        """
        if process:
            if process.poll() is None:  # Check if process is still running
                try:
                    process.terminate()
                    process.wait(timeout=2)  # Add a timeout to avoid indefinite waiting
                    logging.info(f"{name} process terminated successfully.")
                except Exception as e:
                    logging.warning(f"Failed to terminate {name} process gracefully: {e}")
                    process.kill()  # Force terminate if graceful termination fails
                    logging.info(f"{name} process killed.")
            else:
                logging.info(f"{name} process already terminated.")
        else:
            logging.info(f"No {name} process to terminate.")

    def cleanup_resources(self, server_process, safari_process):
        """
        Cleans up resources by terminating the server and Safari processes.
        """
        logging.info("Starting resource cleanup...")
        self.terminate_process(server_process, "Ruffle server")
        self.terminate_process(safari_process, "Safari")
        logging.info("All processes terminated successfully.")