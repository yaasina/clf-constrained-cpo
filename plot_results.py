import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """Calculates the moving average of a 1D array."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def plot_results():
    """
    Loads data for multiple seeds, calculates smoothed rewards and costs,
    and plots them for comparison.
    """
    experiments = [
        # {
        #     "name": "No Scaling",
        #     "rewards_path": "src/result/Pendulum_CPO_1_1_rewards.npz",
        #     "steps_path": "src/result/Pendulum_CPO_1_1_steps.npz",
        #     "costs_path": "src/result/Pendulum_CPO_1_1_costs.npz",
        # },
        {
            "name": "Update b4",
            "rewards_path": "result/Pendulum_CPO_1_19_rewards.npz",
            "steps_path": "result/Pendulum_CPO_1_19_steps.npz",
            "costs_path": "result/Pendulum_CPO_1_19_costs.npz",
        },
        {
            "name": "Scaling Relative to State Norm (makes most sense)",
            "rewards_path": "result/Pendulum_CPO_1_32_rewards.npz",
            "steps_path": "result/Pendulum_CPO_1_32_steps.npz",
            "costs_path": "result/Pendulum_CPO_1_32_costs.npz",
        },
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    window_size = 5

    for exp in experiments:
        try:
            rewards_data = np.load(exp["rewards_path"])
            steps_data = np.load(exp["steps_path"])
            costs_data = np.load(exp["costs_path"])
        except FileNotFoundError as e:
            print(f"Error loading data for {exp['name']}: {e}")
            continue

        rewards = rewards_data[rewards_data.files[0]]
        steps = steps_data[steps_data.files[0]]
        costs = costs_data[costs_data.files[0]]

        episode_costs = []
        last_step = 0
        for step in steps:
            episode_costs.append(np.sum(costs[last_step:step]))
            last_step = step

        if len(rewards) >= window_size:
            smoothed_rewards = moving_average(rewards, window_size)
            smoothed_episode_costs = moving_average(episode_costs, window_size)
            smoothed_steps = steps[window_size - 1:20]

            ax1.plot(smoothed_steps, smoothed_rewards[:16], linestyle='-', label=f'{exp["name"]}')
            ax2.plot(smoothed_steps, smoothed_episode_costs[:16], linestyle='-', label=f'{exp["name"]}')
        else:
            print(f"Not enough data points to apply smoothing for {exp['name']} with window size {window_size}.")

    ax1.set_title('Smoothed Reward per Episode vs. Cumulative Steps')
    ax1.set_xlabel('Cumulative Steps')
    ax1.set_ylabel('Smoothed Reward')
    ax1.grid(True)
    ax1.legend()

    ax2.set_title('Smoothed Cumulative Cost per Episode vs. Cumulative Steps')
    ax2.set_xlabel('Cumulative Steps')
    ax2.set_ylabel('Smoothed Cumulative Cost')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results_comparison.png')
    print("Comparison plots saved to results_comparison.png")

if __name__ == '__main__':
    plot_results()
