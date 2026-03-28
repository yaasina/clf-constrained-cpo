import matplotlib.pyplot as plt
import numpy as np
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

cpo_rew = []
relaxed_rew = []

for i in range(1,11):
    cpo_raw = np.load(os.path.join(curr_dir,"src", "result", "Pendulum_CPO_" + str(i) + "_rewards.npz"))
    cpo_rew.append(cpo_raw[cpo_raw.files[0]])
    relaxed_raw = np.load(os.path.join(curr_dir, "src", "result", "Pendulum_CPO_relaxed_" + str(i) + "_rewards.npz"))
    relaxed_rew.append(relaxed_raw[relaxed_raw.files[0]])

cpo_mean = np.mean(cpo_rew, axis=0)
cpo_std = np.std(cpo_rew, axis=0)
relaxed_mean = np.mean(relaxed_rew, axis=0)
relaxed_std = np.std(relaxed_rew, axis=0)


# Function to compute the moving average
def moving_average(data, window_size=50):
    return [np.mean(data[np.max(i-window_size, 0): i]) for i in range(len(data))] 

# Compute the moving averages
cpo_mean = moving_average(cpo_mean, window_size=50)
cpo_std = moving_average(cpo_std, window_size=50)
relaxed_mean = moving_average(relaxed_mean, window_size=50)
relaxed_std = moving_average(relaxed_std, window_size=50)


# Plot the moving averages with shaded standard deviation
plt.figure(figsize=(10, 6))
x = np.arange(len(cpo_mean))  # X-axis values

# cpo plot with shaded std
plt.plot(x, cpo_mean, label="cpo", color="blue")
plt.fill_between(x, np.array(cpo_mean) - np.array(cpo_std), np.array(cpo_mean) + np.array(cpo_std), color="blue", alpha=0.2)

# relaxed plot with shaded std
plt.plot(x, relaxed_mean, label="relaxed", color="green")
plt.fill_between(x, np.array(relaxed_mean) - np.array(relaxed_std), np.array(relaxed_mean) + np.array(relaxed_std), color="green", alpha=0.2)

# Add labels, title, and legend
plt.legend()
plt.savefig('pendulum_rewards_comparison.png')
# Show the plot
# plt.show()
