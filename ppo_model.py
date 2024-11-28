from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class PPOAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, output_dim)  # Output probabilities for actions
        self.value_head = nn.Linear(128, 1)           # Output state value

    def forward(self, x):
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
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []

    state = env.reset()  # Reset the environment
    for step in range(n_steps):
        # Convert state to a PyTorch tensor
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

        # Forward pass through the policy network
        policy_logits, state_value = policy(state_tensor)
        action_distribution = torch.distributions.Categorical(logits=policy_logits)
        action = action_distribution.sample()  # Sample an action
        log_prob = action_distribution.log_prob(action)  # Log probability of the action

        # Take the action in the environment
        next_state, reward, done, _ = env.step(action.item())

        # Store trajectory data
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(log_prob.item())
        values.append(state_value.item())
        dones.append(done)

        # Prepare for the next step
        state = next_state
        if done:
            state = env.reset()  # Reset on episode completion

    return states, actions, rewards, log_probs, values, dones

def compute_ppo_loss(
    policy, states, actions, rewards, log_probs, values, dones, gamma=0.99, epsilon=0.2
):
    """
    Compute PPO loss (policy, value, and entropy components).

    :param policy: PPO policy network.
    :param states: List of states.
    :param actions: List of actions.
    :param rewards: List of rewards.
    :param log_probs: List of log probabilities of actions.
    :param values: List of value estimates.
    :param dones: List of episode end indicators.
    :param gamma: Discount factor.
    :param epsilon: Clipping parameter for PPO.
    :return: Total loss (policy + value + entropy).
    """
    # Compute returns and advantages
    returns = []
    advantages = []
    running_return = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        running_return = reward + gamma * running_return * (1 - done)
        returns.insert(0, running_return)

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Convert states and actions to tensors
    state_tensor = torch.tensor(states, dtype=torch.float32).view(len(states), -1)
    action_tensor = torch.tensor(actions, dtype=torch.long)

    # Forward pass through the policy
    policy_logits, state_values = policy(state_tensor)
    action_distribution = torch.distributions.Categorical(logits=policy_logits)
    new_log_probs = action_distribution.log_prob(action_tensor)

    # Policy loss (clipped surrogate objective)
    ratios = torch.exp(new_log_probs - torch.tensor(log_probs))
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    # Value loss
    value_loss = (returns - state_values).pow(2).mean()

    # Entropy bonus
    entropy = action_distribution.entropy().mean()

    # Total loss
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
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