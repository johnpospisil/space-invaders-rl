"""
Prioritized Experience Replay Buffer

Implements prioritized experience replay (PER) which samples transitions
based on their TD-error, allowing the agent to learn more from surprising
or important experiences.

Reference: Schaul et al. (2016) - "Prioritized Experience Replay"
"""

import numpy as np
import torch


class SumTree:
    """
    Sum tree data structure for efficient sampling with priorities.
    
    Binary tree where parent nodes store sum of their children's priorities.
    Allows O(log n) updates and O(log n) sampling.
    """
    
    def __init__(self, capacity):
        """
        Initialize sum tree.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Retrieve sample index from tree based on priority sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return total priority sum."""
        return self.tree[0]
    
    def add(self, priority, data):
        """
        Add experience with given priority.
        
        Args:
            priority: Priority value for this experience
            data: Experience tuple (state, action, reward, next_state, done)
        """
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        """
        Update priority for a tree node.
        
        Args:
            idx: Tree index to update
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """
        Get experience based on priority sum.
        
        Args:
            s: Priority sum to search for
            
        Returns:
            (tree_idx, priority, experience)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer using sum tree.
    
    Samples transitions proportionally to their TD-error, with importance
    sampling weights to correct for bias.
    """
    
    def __init__(
        self,
        capacity,
        observation_shape,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        epsilon=1e-6,
        device='cpu'
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            observation_shape: Shape of observations (C, H, W)
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant to prevent zero priorities
            device: Device for tensors
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.device = device
        self.frame = 1
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def _get_beta(self):
        """Get current beta value (annealed from beta_start to 1.0)."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add transition with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """
        Sample batch with prioritized sampling.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        self.frame += 1
        beta = self._get_beta()
        
        # Sample experiences
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            batch.append(experience)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        actions = torch.LongTensor(np.array([exp[1] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp[2] for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([exp[4] for exp in batch])).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: Tree indices to update
            td_errors: TD-errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples."""
        return self.tree.n_entries >= batch_size
    
    def __len__(self):
        """Return current buffer size."""
        return self.tree.n_entries
