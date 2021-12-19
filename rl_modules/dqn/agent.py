import torch
import numpy as np
import random
import collections
from arguments import parse_args

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
constants = parse_args()

# class Agent:
#     def __init__(self, env, exp_buffer, action_range):
#         self.exp_buffer = exp_buffer
#         self.action_range = action_range-1

#     def play_step(self, net, epsilon=0.0, device="cpu"):

#         done_reward = None
#         if np.random.random() < epsilon:
#             action = random.randint(0,self.action_range)
#         else:
#             state_a = np.array([self.state], copy=False)
#             state_v = torch.tensor(state_a).to(device)
#             q_vals_v = net(state_v)
#             _, act_v = torch.max(q_vals_v, dim=1)
#             action = int(act_v.item())
        
#         self.env.step(action)
#         new_state, reward, is_done, _ = self.env.last()
#         self.total_reward += reward*constants.reward_multiplier

#         exp = Experience(self.state, action, reward, is_done, new_state)
#         self.exp_buffer.append(exp)
#         self.state = new_state
#         if is_done:
#             done_reward = self.total_reward
#             self._reset()
#         return done_reward


class Agent:
    def __init__(self, env, exp_buffer, action_range):
        self.exp_buffer = exp_buffer
        self.action_range = action_range-1
    
    def get_action(agent_index, curr_state, epsilons, explore_on=True):
        if np.random.random() < self.epsilons[agent_index] and explore_on:
            selected_action = random.randint(0,self.action_space.n-1)
        else:
            selected_action = self.nets[agent_index]( torch.FloatTensor(curr_state).to(self.device) ).argmax()
            selected_action = selected_action.detach().cpu()
        return selected_action

    def take_step(agent_index, curr_state):
        action = get_action(agent_index,curr_state)
        self.env.step(action)
        return action