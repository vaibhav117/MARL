from pettingzoo.butterfly import knights_archers_zombies_v7
import warnings
warnings.filterwarnings("ignore")
from pettingzoo.utils import random_demo
import numpy as np
import time



env = knights_archers_zombies_v7.env(num_knights=0 , num_archers=2)
env.reset()

print(env.action_spaces)

def run_demo():
    random_demo(env, render=True, episodes=1)

def random_steps(iterations):
    for agent in env.agent_iter():
        print(f"Agent:{agent}")
        observation, reward, done, info = env.last()
        print(done)
        if not done:
            action = 0
        else:
            action = None
        env.step(action)
        env.render(mode='human')

random_steps(10000)
# print(f"Agents: {env.agents}")
# print(f"Action Space: {env.action_spaces}")