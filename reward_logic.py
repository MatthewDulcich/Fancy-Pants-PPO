import logging

def calculate_rewards(
    original_scale_gray_obs, 
    recent_full_res_observations, 
    collected_swirlies, 
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
    wrong_door_penalty = -500
    scale_swirlies_reward = 10

    # Reward for completing the level (door entry)
    if check_for_black_screen_func(original_scale_gray_obs):
        done = True
        logging.info("Black screen detected. Checking which door agent entered...")
        if entered_correct_door_func(recent_full_res_observations):
            logging.info("Finished tutorial level! Reward earned: 500")
            reward += complete_level_reward
        else:
            logging.info("Wrong door entered. Penalty applied: -500")
            reward += wrong_door_penalty

    # Reward for reaching a checkpoint
    checkpoint_reward, checkpoint_id = checkpoint_matching_func(original_scale_gray_obs)
    reward += checkpoint_reward

    # Reward for collecting swirlies
    swirlie_reward = scale_swirlies_reward * collected_swirlies
    reward += swirlie_reward

    # Create info dict for logging and debugging
    info = {
        "swirlies collected": collected_swirlies,
        "swirlie reward": swirlie_reward,
        "swirles detected": collected_swirlies,
        "checkpoint id": checkpoint_id,
        "checkpoint reward": checkpoint_reward,
        "episode reward": reward,
        "done": done,
    }

    return reward, done, info