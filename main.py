from env import Environment
from diagram import Diagram
from config.training_config import TrainingConfig

def main():
    """
    Main function to run the environment, collect collision data, and plot it.
    """
    # 1. Initialize the environment
    # We are using 'SafetyRacecarGoal1-v0' as requested.
    env = Environment(env_id='SafetyRacecarGoal1-v0', render_mode='None')

    # 2. Set up training parameters and data collection
    num_episodes = TrainingConfig.NUM_EPISODES
    max_steps_per_episode = TrainingConfig.MAX_STEPS_PER_EPISODE
    total_collisions = 0
    cumulative_collisions_history = []
    collisions_per_episode = []
    rewards_per_episode = []

    print(f"Running for {num_episodes} episodes...")

    # 3. Run the training loop
    for episode in range(num_episodes):
        env.reset()
        episode_reward = 0
        for step in range(max_steps_per_episode):
            # Using a random agent for demonstration purposes
            action = env.action_space().sample()
            _obs, reward, _cost, terminated, truncated, _info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break
        
        # Update total collisions and record history
        total_collisions += env.collision_count
        collisions_per_episode.append(env.collision_count)
        cumulative_collisions_history.append(total_collisions)
        rewards_per_episode.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f} | Collisions = {env.collision_count} | Total Collisions = {total_collisions}")

    # 4. Plot the results
    diagram = Diagram(cumulative_collisions_history, collisions_per_episode, rewards_per_episode, num_episodes)
    diagram.plot()

if __name__ == '__main__':
    main()
