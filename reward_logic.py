import logging
import config_handler

# Load configuration
config = config_handler.load_config("game_config.json")

# Reward logic parameters
frame_diff_threshold = config["reward_logic"]["frame_diff_threshold"]
positive_frame_diff_scaling_factor = config["reward_logic"]["positive_frame_diff_scaling_factor"]
negative_frame_diff_scaling_factor = config["reward_logic"]["negative_frame_diff_scaling_factor"]
complete_level_reward = config["reward_logic"]["complete_level_reward"]
wrong_door_penalty = config["reward_logic"]["wrong_door_penalty"]
scale_swirlies_reward = config["reward_logic"]["scale_swirlies_reward"]

def calculate_rewards(
    original_scale_gray_obs, 
    recent_full_res_observations, 
    collected_swirlies,
    frame_diff,
    checkpoint_matching_func, 
    check_for_black_screen_func, 
    entered_correct_door_func
):
    """
    Calculate the reward based on game logic.

    Args:
        original_scale_gray_obs (numpy.ndarray): The current grayscale observation.
        recent_full_res_observations (deque): Recent observations for door entry check.
        collected_swirlies (int): Number of swirlies collected.
        checkpoint_matching_func (function): Function to calculate checkpoint rewards.
        check_for_black_screen_func (function): Function to check for black screens.
        entered_correct_door_func (function): Function to check if the correct door was entered.

    Returns:
        reward (int): Total reward calculated.
        done (bool): Whether the episode is done.
        info (dict): Additional information about rewards.
    """
    reward = 0
    done = False

    complete_level_reward = 500
    wrong_door_penalty = 0
    scale_swirlies_reward = 10

    # Reward for completing the level (door entry)
    if check_for_black_screen_func(original_scale_gray_obs):
        done = True
        logging.info("Black screen detected. Checking which door agent entered...")
        if entered_correct_door_func(recent_full_res_observations):
            logging.info(f"Finished tutorial level! Reward earned: {complete_level_reward}")
            reward += complete_level_reward
        else:
            logging.info(f"Wrong door entered. Penalty applied: {wrong_door_penalty}")
            reward += wrong_door_penalty

    # Reward for reaching a checkpoint
    checkpoint_reward, checkpoint_id = checkpoint_matching_func(original_scale_gray_obs)
    reward += checkpoint_reward

    # # Reward for collecting swirlies
    swirlie_reward = scale_swirlies_reward * collected_swirlies
    reward += swirlie_reward

    # Create info dict for logging and debugging
    info = {
        "swirlies collected": collected_swirlies,
        "swirlie reward": swirlie_reward,
        # "swirles detected": collected_swirlies,
        "checkpoint id": checkpoint_id,
        "checkpoint reward": checkpoint_reward,
        "episode reward": reward,
        "done": done,
    }

    return reward, done, info