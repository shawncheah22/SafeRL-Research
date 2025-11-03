import matplotlib.pyplot as plt

class Diagram:
    def __init__(self, cumulative_collisions_history, collisions_per_episode, rewards_per_episode, losses_per_episode):
        self.cumulative_collisions_history = cumulative_collisions_history
        self.collisions_per_episode = collisions_per_episode
        self.rewards_per_episode = rewards_per_episode
        self.losses_per_episode = losses_per_episode

    def plot(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        fig.suptitle('Agent Performance Over Time (SafetyRacecarGoal1-v0)', fontsize=16)
        num_episodes_run = len(self.rewards_per_episode)
        episodes = range(1, num_episodes_run + 1)

        color_cumulative = '#4c72b0'
        color_per_episode = '#dd8452'
        color_reward = '#55a868'
        color_loss = '#c44e52'

        ax1.plot(episodes, self.cumulative_collisions_history, marker='o', linestyle='-', label='Cumulative Collisions', color=color_cumulative)
        ax1.plot(episodes, self.collisions_per_episode, marker='x', linestyle='--', label='Collisions Per Episode', color=color_per_episode)
        ax1.set_title('Collision Metrics')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax2.plot(episodes, self.rewards_per_episode, marker='o', label='Reward Per Episode', color=color_reward)
        ax2.set_title('Reward Per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax3.plot(episodes, self.losses_per_episode, marker='.', linestyle='-', label='Average Loss Per Episode', color=color_loss)
        ax3.set_title('Average Training Loss Per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
