from pettingzoo.butterfly import knights_archers_zombies_v7
import warnings
warnings.filterwarnings("ignore")
from pettingzoo.utils import random_demo
import numpy as np
import time
import supersuit as ss
import matplotlib.pyplot as plt


env = knights_archers_zombies_v7.env(num_knights=0 , num_archers=2)
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env.reset()

print(env.action_spaces)

def run_demo():
    random_demo(env, render=True, episodes=1)

def random_steps(iterations):
    for agent in env.agent_iter():
        print(f"Agent:{agent}")
        obs, reward, done, info = env.last()
        print(done)
        if not done:
            if agent == "archer_0":
                action = 0
            else:
                action = 1
        else:
            action = None
        env.step(action)
        plt.imshow(obs)
        plt.show()
        # env.render(mode='human')


random_steps(10000)
# print(f"Agents: {env.agents}")
# print(f"Action Space: {env.action_spaces}")