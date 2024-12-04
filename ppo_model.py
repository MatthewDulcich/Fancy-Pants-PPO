from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PPOAgent(nn.Module):
    def __init__(self, input_channels, input_height, input_width, output_dim):
        super(PPOAgent, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # Output: 32x(H/4)x(W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # Output: 64x(H/8)x(W/8)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # Output: 64x(H/8)x(W/8)
            nn.ReLU()
        )

        # Compute the flattened size after convolutions
        self.flatten_size = self._get_conv_output_size(input_height, input_width)

        # Fully connected layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Policy and value heads
        self.policy_head = nn.Linear(128, output_dim)  # Output probabilities for actions
        self.value_head = nn.Linear(128, 1)           # Output state value

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Apply weight initialization to all layers in the model.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # He initialization for ReLU
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _get_conv_output_size(self, height, width):
        """
        Compute the size of the flattened output after convolutional layers.
        """
        dummy_input = torch.zeros(1, 1, height, width)  # 1xCxHxW
        conv_output = self.conv_layers(dummy_input)
        return int(torch.prod(torch.tensor(conv_output.shape[1:])).item())  # Flattened size

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(f"Input to PPOAgent must have shape (B, C, H, W), but got {x.shape}")
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.shared_layers(x)
        policy_logits = self.policy_head(x)
        state_value = self.value_head(x)
        return policy_logits, state_value

def collect_rollouts(env, policy, n_steps=2048):
    """
    Collect trajectories (state, action, reward, next_state) for PPO training.

    :param env: FPAGame environment.
    :param policy: PPO policy network.
    :param n_steps: Number of steps to collect per rollout.
    :return: Rollout data (states, actions, rewards, values, log_probs, dones).
    """
    states = deque(maxlen=n_steps)
    actions = deque(maxlen=n_steps)
    rewards = deque(maxlen=n_steps)
    log_probs = deque(maxlen=n_steps)
    values = deque(maxlen=n_steps)
    dones = deque(maxlen=n_steps)

    state = env.reset()  # Reset the environment
    print("Observation shape:", state.shape)
    for step in range(n_steps):
        # Convert state to a PyTorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # normalize the state tensor
        state_tensor = state_tensor / 255.0

        # If the observation is grayscale (1 channel):
        if len(state_tensor.shape) == 3:  # (C, H, W)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension -> (B, C, H, W)
        elif len(state_tensor.shape) == 2:  # (H, W)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions -> (B, C, H, W)
        else:
            raise ValueError(f"Unexpected observation shape: {state_tensor.shape}")

        # Forward pass through the policy network
        policy_logits, state_value = policy(state_tensor)
        action_distribution = torch.distributions.Categorical(logits=policy_logits)
        action = action_distribution.sample()  # Sample an action
        log_prob = action_distribution.log_prob(action)  # Log probability of the action

        # Take the action in the environment
        next_state, reward, done, info = env.step(action.item())

        # Log collected swirlies
        logging.info(
            f"Collected swirlies: {info['swirlies collected']} | Total swirlies: {info['swirlies detected']} | "
            f"Swirles reward: {info['swirlies reward']} | Episode reward: {info['episode reward']} | "
            f"Last 10 rewards: {info['last 10 rewards']} | Action: {action.item()} | Done: {done} | "
            f"Frame Difference: {info['frame difference']} | Total Reward: {info['total reward']}"
        )

        if done:
            state = env.reset()  # Reset on episode completion

        # Store trajectory data
        states.append(state.flatten())  # Preprocessed and flattened
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob.item())
        values.append(state_value.item())
        dones.append(done)

        # Prepare for the next step
        state = next_state
        if done:
            state = env.reset()  # Reset on episode completion

    # Convert deque to numpy arrays for better tensor conversion
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    log_probs = np.array(log_probs)
    values = np.array(values)
    dones = np.array(dones)

    return states, actions, rewards, log_probs, values, dones

def compute_ppo_loss(
    policy, states, actions, rewards, log_probs, values, dones, gamma=0.99, epsilon=0.2, entropy_coef=0.02
):
    """
    Compute PPO loss (policy, value, and entropy components).
    """
    # Compute returns and advantages
    returns = []
    running_return = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        running_return = reward + gamma * running_return * (1 - done)
        returns.insert(0, running_return)

    # Convert to tensors
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)

    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute advantages
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Log advantage stats
    logging.info(
        f"Advantages mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}"
    )

    # Reshape states
    batch_size = len(states)
    state_tensor = torch.tensor(states, dtype=torch.float32).view(batch_size, 1, 150, 200)

    # Forward pass through the policy
    policy_logits, state_values = policy(state_tensor)
    action_distribution = torch.distributions.Categorical(logits=policy_logits)
    new_log_probs = action_distribution.log_prob(actions)

    # Policy loss (clipped surrogate objective)
    ratios = torch.exp(new_log_probs - log_probs_tensor)
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    # Value loss with clipping
    value_loss = (returns - state_values).pow(2).mean()
    clipped_value_loss = torch.clamp(value_loss, 0, 100)  # Prevent excessive value errors

    # Entropy bonus
    entropy = action_distribution.entropy().mean()

    # Log details for debugging
    logging.info(f"Policy logits sample: {policy_logits[:5]}")
    logging.info(f"State values sample: {state_values[:5]}")
    logging.info(f"Actions sample: {actions[:5]}")
    logging.info(f"Returns sample: {returns[:5]}")
    logging.info(f"Values sample: {values[:5]}")
    logging.info(f"Advantages sample: {advantages[:5]}")
    logging.info(f"Log Probs sample: {log_probs_tensor[:5]}")
    logging.info(f"New Log Probs sample: {new_log_probs[:5]}")
    logging.info(f"Ratios mean: {ratios.mean().item():.4f}, std: {ratios.std().item():.4f}")
    logging.info(f"Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} (Clipped: {clipped_value_loss.item():.4f}) | Entropy: {entropy.item():.4f}")
    logging.info(f"Entropy Coefficient: {entropy_coef}")

    # Total loss
    total_loss = policy_loss + 0.5 * clipped_value_loss - entropy_coef * entropy
    return total_loss


def update_policy(policy, optimizer, states, actions, rewards, log_probs, values, dones):
    """
    Update the PPO policy using the collected rollouts.

    :return: Computed PPO loss for logging.
    """
    optimizer.zero_grad()
    loss = compute_ppo_loss(policy, states, actions, rewards, log_probs, values, dones)
    loss.backward()
    optimizer.step()
    return loss.item()  # Return the loss value for logging