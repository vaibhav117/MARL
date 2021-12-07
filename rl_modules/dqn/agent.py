import torch
import numpy as np
import random
import collections

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class Agent:
    def __init__(self, env, exp_buffer, action_range):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.action_range = action_range-1

    def _reset(self):
        self.env.reset()
        self.state , _ , _ , _ = self.env.last()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):

        done_reward = None
        if np.random.random() < epsilon:
            action = random.randint(0,self.action_range)
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        
        self.env.step(action)
        new_state, reward, is_done, _ = self.env.last()
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward