# import mss # Can use mss for screen cap
import pyautogui
import time
import Quartz
# import cv2
import numpy as np
# import pytesseract
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import envs
from gymnasium.spaces import Box, Discrete


class FPAGame(Env):
    # Setup env, action, and obs shapes
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1, 400, 800), dtype=np.uint8)
        self.action_space = Discrete(7) # Number of actions
        self.key_states = {} # Initialize empty key states to keep track of key presses

        # TODO: set the observation window location
        pass

    # Helper function to toggle key presses
    def key_toggle(self, key):
        if key not in self.key_states or not self.key_states[key]:
            pyautogui.keyDown(key)
            self.key_states[key] = True
        else:
            pyautogui.keyUp(key)
            self.key_states[key] = False


    # Called to execute action
    def step(self, action):
        # All other commands - jump_out_right, jump_out_left, jump_up, jump_down, hurricane_attack, spiral_attack, down_attack, up_attack
        # Action Key Mapping
        action_map = {
            0: ['left'],  # left
            1: ['right'],  # right
            2: ['s'],  # jump
            3: ['down'],  # duck, slide, drop_ledge
            4: ['up'],  # up_ledge, open_door
            5: ['a'],  # jab, random_long_attack
            6: ['no_key'],  # do nothing
        }

        if action != 6:
            key_toggle(action_map[action]) # TODO: Fix key_toggle

        # TODO: Get observation, reward, done, etc

        pass
    
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
        pass
    # Get the terminal state
    def get_done(self):
        pass