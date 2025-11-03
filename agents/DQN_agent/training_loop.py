from config.training_config import TrainingConfig
from agents.DQN_agent.DQN_agent import DQNAgent
from agents.DQN_agent.Qnetwork import QNetwork
from agents.DQN_agent.replay_buffer import ReplayBuffer
from agents.DQN_agent.constants import DQNAgentConstants
from env import Environment
from diagram import Diagram
import os

def training_loop():
    """
    Main function to run the DQN agent training loop.
    """
    env = Environment(env_id='SafetyRacecarGoal1-v0', render_mode='none')
    config = DQNAgentConstants()

    # Instantiate the DQNAgent
    agent = DQNAgent(env, config, QNetwork, ReplayBuffer)

    model_filename = f"dqn_checkpoint.pth"
    start_episode = 0
    epsilon = config.EPS_START
    cumulative_collisions_history = []
    collisions_per_episode = []
    rewards_per_episode = []
    
    if os.path.exists(os.path.join("agents/DQN_agent/weights", model_filename)):
        start_episode, epsilon, history_data = agent.load(model_filename)
        cumulative_collisions_history = history_data.get('cumulative_collisions', [])
        collisions_per_episode = history_data.get('collisions_per_episode', [])
        rewards_per_episode = history_data.get('rewards_per_episode', [])
        print(f"Resuming training from episode {start_episode} with epsilon {epsilon:.4f}")
    
    num_episodes = TrainingConfig.NUM_EPISODES
    max_steps_per_episode = TrainingConfig.MAX_STEPS_PER_EPISODE
    
    # Epsilon setup
    eps_decay = config.EPS_DECAY
    eps_end = config.EPS_END
    checkpoint_every = config.CHECKPOINT_EVERY
    
    # Restore total_collisions from the last entry in the cumulative history
    total_collisions = cumulative_collisions_history[-1] if cumulative_collisions_history else 0

    print(f"Running for {num_episodes} episodes...")

    for episode in range(start_episode, num_episodes):
        # Env reset returns the initial state
        state, _ = env.reset() 
        episode_reward = 0.0
        
        for step in range(max_steps_per_episode):
            
            # Agent Select Action (Epsilon-Greedy)
            action_vector, discrete_action_index = agent.select_action(state, epsilon)
            
            # Step the environment
            _obs, reward, _cost, terminated, truncated, _info = env.step(action_vector)
            
            next_state = _obs
            done = terminated or truncated

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

        # Periodically save a checkpoint
        if (episode + 1) % checkpoint_every == 0:
            history_to_save = {
                'cumulative_collisions': cumulative_collisions_history,
                'collisions_per_episode': collisions_per_episode,
                'rewards_per_episode': rewards_per_episode
            }
            agent.save(model_filename, episode, epsilon, history_to_save)

    # Plot the results
    diagram = Diagram(cumulative_collisions_history, collisions_per_episode, rewards_per_episode)
    diagram.plot()
    
    # Save the final model
    final_history = {'cumulative_collisions': cumulative_collisions_history, 'collisions_per_episode': collisions_per_episode, 'rewards_per_episode': rewards_per_episode}
    agent.save(f"dqn_final_episode_{num_episodes}.pth", num_episodes - 1, epsilon, final_history)

if __name__ == '__main__':
    training_loop()
