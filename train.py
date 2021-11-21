from pettingzoo.butterfly import knights_archers_zombies_v7
import warnings
warnings.filterwarnings("ignore")
from pettingzoo.utils import random_demo
import numpy as np
import time
from env import make_env
from rl_modules.dqn.dqn import DQN
from arguments import parse_args

constants = parse_args()

if __name__ == '__main__':
    num_of_agent = constants.num_knights + constants.num_archers
    env = make_env(num_knights=0,num_archers=1)
    env.reset()
    agent_list = env.agents
    agents = []
    print(env.observation_space(agent_list[0]).shape)
    if constants.run_training_flag:
        for _ in range(num_of_agent):
            agents.append( DQN(env.observation_space(agent_list[0]).shape , env.action_space(agent_list[0])) )

