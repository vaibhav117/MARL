from pettingzoo.butterfly import knights_archers_zombies_v7
import warnings
warnings.filterwarnings("ignore")
from pettingzoo.utils import random_demo
import numpy as np
import time
import supersuit as ss
import matplotlib.pyplot as plt
import random

env = knights_archers_zombies_v7.env(spawn_rate=20, num_knights=2 , num_archers=2, killable_knights=False, killable_archers=False, line_death=False)
# env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)
env.reset()

print(env.action_spaces)

def run_demo():
    random_demo(env, render=True, episodes=1)

def random_steps(iterations):
    for agent in env.agent_iter():
        print(f"Agent:{agent}")
        obs, reward, done, info = env.last()
        curr_state = env.state()
        print(done,reward,info)
        if not done:
            if agent == "archer_0":
                action = random.randint(0,5)
            else:
                action = random.randint(0,5)
        else:
            action = None
            
        # plt.imshow(obs)
        # plt.show()

        env.step(action)
        obs1, reward1, done1, info1 = env.last()
        obs2, reward2, done2, info2 = env.last()

        if(obs1.all() != obs2.all()):
            print("env.last() moves time")

        # plt.imshow(curr_state)
        # plt.show()
        env.render(mode='human')
        # time.sleep(0.5)


random_steps(10000)
# print(f"Agents: {env.agents}")
# print(f"Action Space: {env.action_spaces}")