from rl_modules.dqn.dqn import DQN
import torch

import numpy as np
import time

from env import make_env
from arguments import parse_args

from stable_baselines3.dqn import CnnPolicy
# from stable_baselines3 import DQN

constants = parse_args()

class Workspace():
    def __init__(self):
        self.device = torch.device(constants.device)
        self.env = make_env(num_knights=0,num_archers=1)
        self.eval_env = make_env(num_knights=0,num_archers=1)
        self.env.reset()
        self.eval_env.reset()
        self.agent_list = self.env.agents
        self.num_of_agent = constants.num_knights + constants.num_archers
        # self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
        #                                     cfg.replay_buffer_capacity,
        #                                     self.device)
        self.action_space = self.env.action_space(self.agent_list[0])
        self.observation_space = self.env.observation_space(self.agent_list[0])

    def test(self):
        print(f"Action Space: {self.action_space}")
        print(f"Observation Space: {self.observation_space}")

    def train(self):
        dqn = DQN(self.observation_space.shape,self.action_space.shape)


                        

if __name__ == '__main__':
    workspace = Workspace()
    workspace.train()
    if constants.run_training_flag:
        pass
        


