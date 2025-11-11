"""
Replay Buffer for Deep Q-Learning.

Implements experience replay, a key technique that stores and randomly samples
past experiences to break correlation in training data.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    
    def __init__(self, capacity, observation_shape, device='cpu'):
        """
        Initialize a ReplayBuffer object.
        
        Args:
            capacity (int): Maximum size of buffer
            observation_shape (tuple): Shape of observations (e.g., (84, 84, 4))
            device (str): Device to store tensors on
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.device = device
        
        # Pre-allocate memory for efficiency
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def push(self, observation, action, reward, next_observation, done):
        """
        Add a new experience to memory.
        
        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Whether episode ended
        """
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        
        Args:
            batch_size (int): Size of batch to sample
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.size
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size
