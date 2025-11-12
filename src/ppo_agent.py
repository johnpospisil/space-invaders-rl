"""
PPO (Proximal Policy Optimization) Agent Implementation

PPO is a policy gradient method that uses a clipped surrogate objective
to prevent too large policy updates, making training more stable than
vanilla policy gradients.

Reference: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared feature extraction.
    
    Actor: Outputs action probabilities π(a|s)
    Critic: Outputs state value V(s)
    """
    
    def __init__(self, observation_shape, n_actions):
        """
        Initialize Actor-Critic network.
        
        Args:
            observation_shape: Shape of observations (H, W, C)
            n_actions: Number of possible actions
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Handle both (H, W, C) and (C, H, W) formats
        if len(observation_shape) == 3:
            if observation_shape[0] == observation_shape[1]:  # Likely (H, W, C)
                in_channels = observation_shape[2]
            else:  # Likely (C, H, W)
                in_channels = observation_shape[0]
        else:
            raise ValueError(f"Expected 3D observation shape, got {observation_shape}")
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Shared hidden layer
        self.fc_shared = nn.Linear(linear_input_size, 512)
        
        # Actor head (policy)
        self.actor = nn.Linear(512, n_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(512, 1)
    
    def forward(self, x):
        """
        Forward pass through network.
        
        Args:
            x: Input observations (batch_size, H, W, C) or (batch_size, C, H, W)
            
        Returns:
            action_probs: Action probability distribution
            state_values: State values
        """
        # Ensure correct shape (batch, channels, height, width)
        if x.dim() == 4 and x.shape[3] < x.shape[1]:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        
        # Actor output (action probabilities)
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output (state value)
        state_values = self.critic(x)
        
        return action_probs, state_values
    
    def get_action_and_value(self, x):
        """
        Sample action and get value for given state.
        
        Args:
            x: Input observation
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value
        """
        action_probs, value = self.forward(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class RolloutBuffer:
    """
    Buffer for storing trajectories during PPO rollout phase.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, log_prob, reward, value, done):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO agent with clipped surrogate objective.
    """
    
    def __init__(
        self,
        observation_shape,
        n_actions,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=256,
        n_steps=2048,
        device='cpu'
    ):
        """
        Initialize PPO agent.
        
        Args:
            observation_shape: Shape of observations
            n_actions: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of optimization epochs per rollout
            batch_size: Mini-batch size for updates
            n_steps: Number of steps to collect per rollout
            device: Device to run on
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.device = device
        
        # Create network
        self.network = ActorCriticNetwork(observation_shape, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
    
    def select_action(self, state, training=True):
        """
        Select action using current policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode
            
        Returns:
            action: Selected action
            log_prob: Log probability (if training)
            value: State value (if training)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
                return action.item(), log_prob.item(), value.item()
            else:
                action_probs, _ = self.network(state_tensor)
                action = action_probs.argmax(1).item()
                return action
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            next_value: Value of final next state
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self):
        """
        Perform PPO update using collected rollout data.
        
        Returns:
            Dictionary of training metrics
        """
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        values = self.buffer.values
        
        # Compute advantages and returns
        with torch.no_grad():
            next_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            _, next_value = self.network(next_state)
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(
            self.buffer.rewards, values, self.buffer.dones, next_value
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.n_epochs):
            # Create mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get current policy and value
                action_probs, state_values = self.network(states[batch_indices])
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions[batch_indices])
                entropy = dist.entropy().mean()
                
                # Compute ratio and clipped surrogate
                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_indices]
                
                # Policy loss (maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(state_values.squeeze(), returns[batch_indices])
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
    
    def save(self, path):
        """Save agent state."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
