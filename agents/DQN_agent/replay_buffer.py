from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store standard DQN experience tuples."""

    def __init__(self, buffer_size: int, device=torch.device("cpu")):
        """Initialize a ReplayBuffer object."""
        self.memory = deque(maxlen=buffer_size)
        self.device = device

    # Updated push method for the standard 5-tuple
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory (s, a, r, s', d)."""
        # Store the 5-element tuple
        self.memory.append((state, action, reward, next_state, done))
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory and convert to tensors."""
        experiences = random.sample(self.memory, k=batch_size)

        # Use zip(*) to efficiently separate the list of 5-tuple experiences 
        # into five separate lists (states, actions, rewards, etc.)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert lists to PyTorch tensors and move them to the device
        
        # States and Next States are float tensors
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        
        # Actions are long tensors (indices)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        
        # Rewards are float tensors (rewards can be fractions)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        
        # Dones are float tensors (0 or 1 for multiplication in the Q-target calculation)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)