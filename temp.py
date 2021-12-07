import matplotlib.pyplot as plt
import numpy as np
import math

episode_count = 10000

rewards = np.random.rand(episode_count)


episode_10 = int(episode_count/10)
episode_100 = int(episode_count/100)
episode_1000 = int(episode_count/1000)

per_episode_noise = np.random.normal(0,0.4,episode_count)
per_10_episode_noise = np.random.normal(0,0.1,episode_10)
per_100_episode_noise = np.random.normal(0,0.1,episode_100)
per_1000_episode_noise = np.random.normal(0,0.1,episode_1000)
per_1000_episode_noise_2 = np.random.normal(0,0.2,episode_1000)

min_rewards = 0.0
max_rewards = 10.0

for index in range(episode_count):
    index_10 = int(index/10)
    index_100 = int(index/100)
    index_1000 = int(index/1000)
    rewards[index] = index*(max_rewards-min_rewards)/episode_count + per_episode_noise[index] + per_10_episode_noise[index_10] + per_100_episode_noise[index_100] + per_1000_episode_noise[index_1000] + per_1000_episode_noise_2[index_1000]
    rewards[index] = 10*math.tanh(index/4000) + per_10_episode_noise[index_10] + per_100_episode_noise[index_100] + per_1000_episode_noise[index_1000] + per_1000_episode_noise_2[index_1000]


plt.plot(rewards)
plt.show()