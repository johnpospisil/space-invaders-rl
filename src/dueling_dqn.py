"""
Dueling DQN Network Implementation

Dueling networks separate the value function V(s) and advantage function A(s,a):
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

This improves learning by explicitly separating state values from action advantages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    
    Reference: Wang et al. (2016) - "Dueling Network Architectures for Deep RL"
    """
    
    def __init__(self, observation_shape, n_actions):
        """
        Initialize Dueling DQN network.
        
        Args:
            observation_shape: Shape of input observations (H, W, C) or (C, H, W)
            n_actions: Number of possible actions
        """
        super(DuelingDQN, self).__init__()
        
        # Handle both (H, W, C) and (C, H, W) formats
        if len(observation_shape) == 3:
            if observation_shape[0] == observation_shape[1]:  # Likely (H, W, C)
                in_channels = observation_shape[2]
                height = observation_shape[0]
                width = observation_shape[1]
            else:  # Likely (C, H, W)
                in_channels = observation_shape[0]
                height = observation_shape[1]
                width = observation_shape[2]
        else:
            raise ValueError(f"Expected 3D observation shape, got {observation_shape}")
        
        # Shared convolutional layers (feature extraction)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single value output
        )
        
        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)  # One advantage per action
        )
    
    def forward(self, x):
        """
        Forward pass through dueling network.
        
        Args:
            x: Input observations (batch_size, C, H, W) or (batch_size, H, W, C)
            
        Returns:
            Q-values (batch_size, n_actions)
        """
        # Ensure correct shape (batch, channels, height, width)
        if x.dim() == 4 and x.shape[1] != x.shape[2]:
            # Already in correct format (B, C, H, W)
            pass
        elif x.dim() == 4 and x.shape[3] < x.shape[1]:
            # Likely (B, H, W, C) - permute to (B, C, H, W)
            x = x.permute(0, 3, 1, 2).contiguous()
        
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view for safety
        
        # Split into value and advantage streams
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine using dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class DoubleDuelingDQNAgent:
    """
    Agent combining Double DQN and Dueling DQN improvements.
    
    - Double DQN: Reduces overestimation by using online network for action selection
    - Dueling DQN: Separates value and advantage estimation
    """
    
    def __init__(
        self,
        observation_shape,
        n_actions,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9999,
        device='cpu'
    ):
        """
        Initialize Double Dueling DQN agent.
        
        Args:
            observation_shape: Shape of observations (C, H, W)
            n_actions: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per step
            device: Device to run on ('cpu' or 'cuda')
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        # Create online and target networks (Dueling architecture)
        self.online_network = DuelingDQN(observation_shape, n_actions).to(device)
        self.target_network = DuelingDQN(observation_shape, n_actions).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation (H, W, C) or (C, H, W)
            training: Whether in training mode (use epsilon-greedy)
            
        Returns:
            Selected action index
        """
        if training and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Ensure correct shape (batch, channels, height, width)
            if state_tensor.dim() == 4 and state_tensor.shape[3] < state_tensor.shape[1]:
                state_tensor = state_tensor.permute(0, 3, 1, 2)
            q_values = self.online_network(state_tensor)
            return q_values.argmax(1).item()
    
    def train_step(self, batch):
        """
        Perform one training step using Double DQN update rule.
        
        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
            
        Returns:
            Training loss value
        """
        states, actions, rewards, next_states, dones = batch
        
        # Current Q-values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using online network
            next_actions = self.online_network(next_states).argmax(1, keepdim=True)
            # Evaluate those actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network with online network weights."""
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def save(self, path):
        """Save agent state."""
        torch.save({
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
