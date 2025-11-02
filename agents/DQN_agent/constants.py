from dataclasses import dataclass

@dataclass(frozen=True)
class DQNAgentConstants: 
    """Constants for the DQN Agent."""

    OBS_STATE_SIZE = 60
    ACTION_SIZE = 25
    BATCH_SIZE = 64

    GAMMA = 0.95  # Discount factor for future rewards
    TAU = 0.001  # For soft update of target parameters
    LEARNING_RATE = 0.0001  # Learning rate for the optimizer
    UPDATE_EVERY_STEPS = 4  # How often to update the network

    CHECKPOINT_EVERY = 100

    # Epsilon-Greedy Strategy
    EPS_START = 1.0         # Starting value of epsilon
    EPS_END = 0.01          # Minimum value of epsilon
    EPS_DECAY = 0.995       # Multiplicative factor for epsilon decay