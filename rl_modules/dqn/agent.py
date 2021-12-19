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