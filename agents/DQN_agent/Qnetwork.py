import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, obs_size = 48, action_size = 25):
        """
        Q-Network for Deep Q-Learning (DQN).
        Takes a state as input and outputs Q-values for all discrete actions.
        Environment has continuous action space, read AgentReadMe.md for discretisation details.
        """
        super().__init__()

        # Define network layers 
        self.fc1 = nn.Linear(obs_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        """
        Passes a state through the network to compute Q-values.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

