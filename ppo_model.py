from collections import deque
import torch.nn as nn
import logging
from collections import deque
import torch
from torch.distributions import Categorical
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class PPOAgent(nn.Module):
    def __init__(self, input_channels, input_height, input_width, n_actions):
        """
        A simple PPO agent with convolutional and linear layers.
        
        :param input_channels: Number of input channels (e.g., grayscale: 1, RGB: 3).
        :param input_height: Height of the input image.
        :param input_width: Width of the input image.
        :param n_actions: Number of possible actions.
        """
        super(PPOAgent, self).__init__()

        # Convolutional layers for processing image input
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, H/8, W/8)
            nn.ReLU(),
        )

        # Calculate flattened size after convolutional layers
        conv_output_height = input_height // 8
        # if input_height % 8 != 0:
        #     conv_output_height += 1
        conv_output_width = input_width // 8
        flattened_size = 64 * conv_output_height * conv_output_width

        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
        )

        # Policy head (output action probabilities)
        self.policy_head = nn.Linear(128, n_actions)

        # Value head (output state value)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass through the PPO agent.
        
        :param x: Input tensor (B, C, H, W), where B is batch size, C is channels, H and W are height and width.
        :return: Tuple of (policy logits, state value).
        """
        # Pass through convolutional layers
        x = self.conv_layers(x)  # Output shape: (B, 64, H/8, W/8)

        # Flatten the output for the linear layers
        x = x.view(x.size(0), -1)  # Output shape: (B, flattened_size)

        # Pass through shared fully connected layers
        x = self.shared_fc(x)

        # Get policy logits and state value
        policy_logits = self.policy_head(x)
        state_value = self.value_head(x)

        return policy_logits, state_value

def collect_rollouts(env, policy, n_steps=2048, gamma=0.99, max_buffer_size=10000):
    """
    Collect trajectories (state, action, reward, next_state) for PPO training, using deque for efficiency.

    :param env: The environment to interact with.
    :param policy: The PPOAgent.
    :param n_steps: Number of steps to collect in each rollout.
    :param gamma: Discount factor for future rewards.
    :param max_buffer_size: Maximum size for the deque buffer.
    :return: Rollout data (states, actions, rewards, log_probs, values, dones).
    """
    # Use deque for efficient sequential operations
    states = deque(maxlen=max_buffer_size)
    actions = deque(maxlen=max_buffer_size)
    rewards = deque(maxlen=max_buffer_size)
    log_probs = deque(maxlen=max_buffer_size)
    values = deque(maxlen=max_buffer_size)
    dones = deque(maxlen=max_buffer_size)

    state = env.reset()  # Reset the environment and get the initial state

    for step in range(n_steps):
        # Normalize the state and reshape for the PPOAgent
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension (B=1)
        state_tensor = state_tensor / 255.0  # Normalize pixel values to [0, 1]

        # Forward pass through the policy to get action probabilities and state value
        policy_logits, state_value = policy(state_tensor)
        action_distribution = Categorical(logits=policy_logits)
        action = action_distribution.sample()  # Sample an action
        log_prob = action_distribution.log_prob(action)  # Log probability of the action

        # Take the action in the environment
        next_state, reward, done, _ = env.step(action.item())

        # Append rollout data to deque
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob.item())
        values.append(state_value.item())
        dones.append(done)

        # Update the current state
        state = next_state

        # Break the loop if the episode ends
        if done:
            state = env.reset()

    # Convert deque to tensors for efficient batch processing
    states = torch.tensor(np.array(states), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32)
    values = torch.tensor(np.array(values), dtype=torch.float32)
    dones = torch.tensor(np.array(dones), dtype=torch.float32)

    return states, actions, rewards, log_probs, values, dones

def compute_ppo_loss(policy, states, actions, rewards, log_probs, values, dones, gamma=0.99, epsilon=0.2, entropy_coef=0.01):
    """
    Compute PPO loss including policy loss, value loss, and entropy bonus.

    :param policy: The PPOAgent.
    :param states: Collected states (Tensor).
    :param actions: Collected actions (Tensor).
    :param rewards: Collected rewards (Tensor).
    :param log_probs: Log probabilities of actions from the old policy (Tensor).
    :param values: State values predicted by the old policy (Tensor).
    :param dones: Episode termination flags (Tensor).
    :param gamma: Discount factor for future rewards.
    :param epsilon: Clipping parameter for PPO.
    :param entropy_coef: Coefficient for the entropy bonus.
    :return: Total loss (policy loss + value loss - entropy bonus).
    """
    # Step 1: Compute returns (discounted rewards) and advantages
    returns = []
    discounted_sum = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted_sum = 0
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = returns - values.detach()  # Advantage function: A(s, a) = R - V(s)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

    # Step 2: Forward pass through the policy to get new log probs and values
    policy_logits, state_values = policy(states)
    action_distribution = Categorical(logits=policy_logits)
    new_log_probs = action_distribution.log_prob(actions)
    entropy = action_distribution.entropy().mean()

    # Step 3: Compute the policy loss using the clipped objective
    ratios = torch.exp(new_log_probs - log_probs)  # Importance sampling ratios
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    # Step 4: Compute the value loss (squared difference between returns and values)
    value_loss = (returns - state_values).pow(2).mean()

    # Step 5: Combine all losses
    total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

    # Logging for debugging purposes
    logging.info(f"Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} | Entropy: {entropy.item():.4f}")

    # Log details for debugging
    logging.info(f"Policy logits sample: {policy_logits[:5]}")
    logging.info(f"State values sample: {state_values[:5]}")
    logging.info(f"Actions sample: {actions[:5]}")
    logging.info(f"Returns sample: {returns[:5]}")
    logging.info(f"Values sample: {values[:5]}")
    logging.info(f"Advantages sample: {advantages[:5]}")
    logging.info(f"Log Probs sample: {log_probs[:5]}")
    logging.info(f"New Log Probs sample: {new_log_probs[:5]}")
    logging.info(f"Ratios mean: {ratios.mean().item():.4f}, std: {ratios.std().item():.4f}")
    logging.info(f"Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} | Entropy: {entropy.item():.4f}")
    logging.info(f"Entropy Coefficient: {entropy_coef}")

    return total_loss

def update_policy(policy, optimizer, states, actions, rewards, log_probs, values, dones, gamma=0.99, epsilon=0.2, entropy_coef=0.01):
    """
    Update the PPO policy using the collected rollouts.

    :param policy: The PPOAgent.
    :param optimizer: The optimizer for updating policy parameters.
    :param states: Collected states (Tensor).
    :param actions: Collected actions (Tensor).
    :param rewards: Normalized rewards (Tensor).
    :param log_probs: Log probabilities of actions from the old policy (Tensor).
    :param values: State values predicted by the old policy (Tensor).
    :param dones: Episode termination flags (Tensor).
    :param gamma: Discount factor for future rewards.
    :param epsilon: Clipping parameter for PPO.
    :param entropy_coef: Coefficient for the entropy bonus.
    :return: PPO loss value for logging.
    """
    optimizer.zero_grad()  # Reset gradients
    # Compute PPO loss
    loss = compute_ppo_loss(
        policy=policy,
        states=states,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        dones=dones,
        gamma=gamma,
        epsilon=epsilon,
        entropy_coef=entropy_coef
    )
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the policy parameters

    logging.info(f"PPO Loss: {loss.item():.4f}")
    return loss.item()  # Return the loss value for logging