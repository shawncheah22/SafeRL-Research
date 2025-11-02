from diagram import Diagram
from env import Environment
from config.training_config import TrainingConfig
import numpy as np
from agents.DQN_agent.DQN_agent import DQNAgent
from agents.DQN_agent.Qnetwork import QNetwork
from agents.DQN_agent.replay_buffer import ReplayBuffer
from agents.DQN_agent.constants import DQNAgentConstants
from env import Environment
from diagram import Diagram

def training_loop():
    """
    Main function to run the DQN agent training loop.
    """
    env = Environment(env_id='SafetyRacecarGoal1-v0', render_mode='None')
    config = DQNAgentConstants()

    # Instantiate the DQNAgent
    agent = DQNAgent(env, config, QNetwork, ReplayBuffer)
    
    num_episodes = TrainingConfig.NUM_EPISODES
    max_steps_per_episode = TrainingConfig.MAX_STEPS_PER_EPISODE
    
    # Epsilon setup
    epsilon = config.EPS_START
    eps_decay = config.EPS_DECAY
    eps_end = config.EPS_END
    
    # Data collection
    total_collisions = 0
    cumulative_collisions_history = []
    collisions_per_episode = []
    rewards_per_episode = []

    print(f"Running for {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Env reset returns the initial state
        state, _ = env.reset() 
        episode_reward = 0.0
        
        for step in range(max_steps_per_episode):
            
            # Agent Select Action (Epsilon-Greedy)
            action_vector = agent.select_action(state, epsilon)
            
            # Step the environment
            _obs, reward, _cost, terminated, truncated, _info = env.step(action_vector)
            
            next_state = _obs
            done = terminated or truncated
            
            
            # 1. Find the indices of the continuous values
            speed_val = action_vector[0]
            steer_val = action_vector[1]
            
            # The .index() call finds the position of the value in the list
            speed_index = agent.speed_actions.index(speed_val)
            steer_index = agent.steer_actions.index(steer_val)
            
            discrete_action_index = (speed_index * len(agent.steer_actions)) + steer_index
            
            # The agent.update() expects the discrete action index.
            agent.update(state, discrete_action_index, reward, next_state, done)
            
            state = next_state # Move to the next state
            episode_reward += float(reward)

            if done:
                break
        
        # Decrease epsilon after the episode ends
        epsilon = max(eps_end, epsilon * eps_decay)
        
        # Update total collisions and record history
        total_collisions += env.collision_count
        collisions_per_episode.append(env.collision_count)
        cumulative_collisions_history.append(total_collisions)
        rewards_per_episode.append(episode_reward)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f} | Collisions = {env.collision_count} | Total Collisions = {total_collisions} | Epsilon = {epsilon:.4f}")

    # 4. Plot the results
    diagram = Diagram(cumulative_collisions_history, collisions_per_episode, rewards_per_episode, num_episodes)
    diagram.plot()
    
    # 5. Save the final model
    agent.save(f"dqn_final_episode_{num_episodes}.pth")

if __name__ == '__main__':
    training_loop()
