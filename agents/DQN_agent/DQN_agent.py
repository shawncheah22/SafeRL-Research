import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
from agents.DQN_agent.Qnetwork import QNetwork
from agents.base_agent import BaseAgent
from agents.DQN_agent.constants import DQNAgentConstants
from agents.DQN_agent.replay_buffer import ReplayBuffer
import random
import os

class DQNAgent(BaseAgent):
    """
    DQN Agent that implements epsilon-greedy action selection and Q-learning updates.
    """
    def __init__(self, env, config: DQNAgentConstants, QNetworkClass=QNetwork, ReplayBufferClass=ReplayBuffer):
        super().__init__(env, config)

        self.state_size = config.OBS_STATE_SIZE
        self.action_size = config.ACTION_SIZE
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.learning_rate = config.LEARNING_RATE
        self.update_every_steps = config.UPDATE_EVERY_STEPS

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Local Q-Network
        self.qnetwork_local = QNetworkClass(self.state_size, self.action_size).to(self.device)

        # Target Q-Network
        self.qnetwork_target = QNetworkClass(self.state_size, self.action_size).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayBufferClass(buffer_size=100000, device=self.device)
        self.t_step = 0

        # Discretization mapping (needed for action selection)
        self.speed_actions = [-5.0, 0.0, 5.0, 10.0, 15.0] # Slower speeds for better control
        self.steer_actions = [-0.5, -0.25, 0.0, 0.25, 0.5] # Finer steering angles (~28.6, ~14.3 degrees)

        # The agent will output an index 0-24, which needs mapping to the continuous space 
            
    def select_action(self, state, episilon):
        """
        Returns a continuous action for the environment and the discrete action index
        for learning, following an epsilon-greedy policy.
        """
        # convert state to tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval() # Set network to evaluation mode
        q_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # Set network back to training mode

        # Epsilon-greedy action selection
        if np.random.rand() > episilon:
            # Select the action with highest Q-value
            discrete_action_index = q_values.argmax(dim=1).item()
        else:
            discrete_action_index = random.choice(np.arange(self.action_size))

        # Mapping discrete action index to continuous action
        speed_index = discrete_action_index // len(self.steer_actions) # len is 5
        steer_index = discrete_action_index % len(self.steer_actions) # len is 5

        speed = self.speed_actions[speed_index]
        steering = self.steer_actions[steer_index]

        continuous_action = np.array([speed, steering], dtype=np.float64)

        return continuous_action, discrete_action_index

    def update(self, state, action, reward, next_state, done): 
        '''
        Stores the transition and checks if it's time to learn.
        Returns the loss if a learning step was performed, otherwise None.
        '''
        loss = None
        # Store experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every_steps

        # Check if conditions are met to learn 
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            loss = self.learn(experiences, self.gamma)
        return loss

    def save(self, filename: str, episode: int, epsilon: float, history_data: dict):
        """
        Save the agent's checkpoint (model, optimizer, episode, epsilon, history) to a file.
        """
        save_dir = "agents/DQN_agent/weights"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, filename)

        checkpoint = {
            'episode': episode,
            'epsilon': epsilon,
            'model_state_dict': self.qnetwork_local.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history_data': history_data
        }

        torch.save(checkpoint, full_path)
        print(f"Checkpoint saved to: {full_path}")

    def load(self, filename):
        """
        Load the agent's model(s) from a file inside the 'weights/' folder.
        Initializes both local and target networks with the saved weights.
        """
        load_dir = "agents/DQN_agent/weights"
        full_path = os.path.join(load_dir, filename)

        if not os.path.exists(full_path):
            print(f"Error: Model file not found at {full_path}")
            # Return defaults if no file
            default_history = {
                'cumulative_collisions': [],
                'collisions_per_episode': [],
                'rewards_per_episode': [],
                'losses_per_episode': []
            }
            return 0, self.config.EPS_START, default_history
            
        checkpoint = torch.load(full_path, map_location=self.device)

        # Apply the weights to the Local Network
        self.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        
        # Apply the weights to the Target Network (synchronize them)
        self.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_episode = checkpoint.get('episode', 0) + 1
        epsilon = checkpoint.get('epsilon', self.config.EPS_START)
        history_data = checkpoint.get('history_data', {
            'cumulative_collisions': [],
            'collisions_per_episode': [],
            'rewards_per_episode': [],
            'losses_per_episode': []
        })

        print(f"Checkpoint loaded from: {full_path}. Resuming from episode {start_episode}.")
        print(f"Loaded history with {len(history_data['rewards_per_episode'])} entries.")

        return start_episode, epsilon, history_data

    def learn(self, experiences, gamma):
        '''
        Update value parameters using given batch of experience tuples.
        '''
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next state) from target model
        q_values_next_state_from_target = self.qnetwork_target(next_states).detach().max(dim=1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_values_next_state_from_target * (1 - dones))

        # Get expected Q values from local network
        q_expected = self.qnetwork_local(states).gather(dim=1, index=actions)

        # Compute MSE loss for local Q-network
        local_network_loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        local_network_loss.backward()
        self.optimizer.step()

        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return local_network_loss.item()

    def _soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)