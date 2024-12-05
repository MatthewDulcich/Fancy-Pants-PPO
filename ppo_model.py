import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
from logging_config import configure_logging

# Configure logging
logging, _ = configure_logging()  # Ignore the log filename if not needed here

# Example usage in ppo_model.py
logging.info("PPO model initialized.")

class PPOAgent(nn.Module):
    def __init__(self, input_channels, input_height, input_width, n_actions):
        super(PPOAgent, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        conv_output_height = input_height // 8
        conv_output_width = input_width // 8
        flattened_size = 64 * conv_output_height * conv_output_width

        self.shared_fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        policy_logits = self.policy_head(x)
        state_value = self.value_head(x)
        return policy_logits, state_value


class PPO:
    def __init__(self, input_channels, input_height, input_width, n_actions, lr=1e-4, gamma=0.98, epsilon=0.3, entropy_coef=0.02):
        self.policy = PPOAgent(input_channels, input_height, input_width, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef

    def collect_rollouts(self, env, n_steps=2048, max_buffer_size=10000):
        states = deque(maxlen=max_buffer_size)
        actions = deque(maxlen=max_buffer_size)
        rewards = deque(maxlen=max_buffer_size)
        log_probs = deque(maxlen=max_buffer_size)
        values = deque(maxlen=max_buffer_size)
        dones = deque(maxlen=max_buffer_size)

        state = env.reset()

        for _ in range(n_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) / 255.0
            policy_logits, state_value = self.policy(state_tensor)
            action_dist = Categorical(logits=policy_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(state_value.item())
            dones.append(done)

            state = next_state if not done else env.reset()

        return (
            torch.tensor(np.array(states), dtype=torch.float32) / 255.0,
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(log_probs), dtype=torch.float32),
            torch.tensor(np.array(values), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def compute_ppo_loss(self, states, actions, rewards, log_probs, values, dones):
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted_sum = reward + self.gamma * discounted_sum * (1 - done)
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_logits, state_values = self.policy(states)
        action_dist = Categorical(logits=policy_logits)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()

        ratios = torch.exp(new_log_probs - log_probs)
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        value_loss = (returns - state_values).pow(2).mean()

        # Log calculations
        logging.info("=== PPO Loss Calculation ===")
        logging.info(f"Returns: {returns[:5]}...")
        logging.info(f"Advantages: {advantages[:5]}...")
        logging.info(f"Policy Ratios: {ratios.mean().item():.4f}")
        logging.info(f"Clipped Ratios: {clipped_ratios.mean().item():.4f}")
        logging.info(f"Policy Loss: {policy_loss.item():.4f}")
        logging.info(f"Value Loss: {value_loss.item():.4f}")
        logging.info(f"Entropy: {entropy.item():.4f}")
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        logging.info(f"Total Loss: {total_loss.item():.4f}")

        return total_loss

    def update_policy(self, states, actions, rewards, log_probs, values, dones, k_epochs=4, clip_grad=0.5):
        for epoch in range(k_epochs):
            self.optimizer.zero_grad()
            loss = self.compute_ppo_loss(states, actions, rewards, log_probs, values, dones)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=clip_grad)
            self.optimizer.step()
            if epoch == 0 or epoch == k_epochs - 1:
                logging.info(f"Epoch {epoch + 1}/{k_epochs} | PPO Loss: {loss.item():.4f}")
            
        return loss.item()