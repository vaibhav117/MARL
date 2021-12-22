import torch
import numpy as np
import random
import collections
from arguments import parse_args

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
constants = parse_args()

"""
This is the Agent class. Agents are responsible 
for interacting with the Env to take actions. 
"""

class Agent:
    def __init__(self, env, exp_buffer, action_range, device):
        self.env = env
        self.exp_buffer = exp_buffer
        self.action_range = action_range-1
        self.device = device
    
    def get_action(self, agent_index, net, curr_state, epsilons, explore_on=True):
        if np.random.random() < epsilons[agent_index] and explore_on:
            selected_action = random.randint(0,self.action_range)
        else:
            state = np.array([curr_state], copy=False)
            selected_action = net( torch.FloatTensor(state).to(self.device) ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def take_step(self, agent_index, net, curr_state, epsilons, explore_on=True):
        action = self.get_action(agent_index, net, curr_state, epsilons, explore_on=explore_on)
        self.env.step(action)
        return action