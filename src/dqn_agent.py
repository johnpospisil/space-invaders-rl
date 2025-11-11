"""
Deep Q-Network (DQN) implementation.

Implements the DQN algorithm from "Playing Atari with Deep Reinforcement Learning"
and "Human-level control through deep reinforcement learning" (Nature paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    """
    Convolutional Neural Network for Q-value approximation.
    Architecture from DQN Nature paper.
    """
    
    def __init__(self, input_shape, n_actions):
        """
        Initialize DQN network.
        
        Args:
            input_shape (tuple): Shape of input (C, H, W), e.g., (4, 84, 84)
            n_actions (int): Number of possible actions
        """
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Convolutional layers (same as Nature DQN paper)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size of flattened features
        conv_out_size = self._get_conv_output_size(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def _get_conv_output_size(self, shape):
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()))
    
    def forward(self, x):
        """
        Forward pass through network.
        
        Args:
            x: Input tensor of shape (batch_size, C, H, W)
            
        Returns:
            Q-values for each action
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class DQNAgent:
    """
    DQN Agent with epsilon-greedy exploration and experience replay.
    """
    
    def __init__(
        self,
        observation_shape,
        n_actions,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        device='cpu'
    ):
        """
        Initialize DQN Agent.
        
        Args:
            observation_shape: Shape of observations (H, W, C)
            n_actions: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            device: Device to run on ('cpu' or 'cuda')
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)
        
        # Convert observation shape from (H, W, C) to (C, H, W) for PyTorch
        input_shape = (observation_shape[2], observation_shape[0], observation_shape[1])
        
        # Create Q-network and target network
        self.q_network = DQN(input_shape, n_actions).to(self.device)
        self.target_network = DQN(input_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Tracking
        self.training_step = 0
    
    def select_action(self, observation, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            training: Whether in training mode (uses epsilon-greedy if True)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(self.n_actions)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                obs_tensor = self._prepare_observation(observation)
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()
    
    def _prepare_observation(self, observation):
        """Convert observation to PyTorch tensor with correct shape."""
        # Convert from (H, W, C) to (1, C, H, W)
        obs = np.transpose(observation, (2, 0, 1))
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return obs_tensor
    
    def train_step(self, batch):
        """
        Perform one training step.
        
        Args:
            batch: Tuple of (observations, actions, rewards, next_observations, dones)
            
        Returns:
            Loss value
        """
        observations, actions, rewards, next_observations, dones = batch
        
        # Convert to PyTorch tensors and change shape from (B, H, W, C) to (B, C, H, W)
        observations = torch.FloatTensor(np.transpose(observations, (0, 3, 1, 2))).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_observations = torch.FloatTensor(np.transpose(next_observations, (0, 3, 1, 2))).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_observations).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (Huber loss is more robust than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
