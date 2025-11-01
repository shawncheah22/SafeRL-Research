
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    An abstract base class for all reinforcement learning agents.
    """
    def __init__(self, env, config):
        """
        Initializes the agent.
        
        Args:
            env: The environment the agent will interact with.
            config: A configuration object with hyperparameters.
        """
        self.env = env
        self.config = config

    @abstractmethod
    def select_action(self, state):
        """
        Given the current state, select an action.
        """
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's model based on a transition from the environment.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Save the agent's learned model(s) to a file.
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Load the agent's learned model(s) from a file.
        """
        pass
