from rl_modules.dqn.dqn import DQN
import torch
import torch.optim as optim
import torch.nn as nn 

import numpy as np
import collections
import os
from MakeTreeDir import MAKETREEDIR
import time
import wandb
from datetime import datetime

from env import make_env
from arguments import parse_args
from rl_modules.dqn.agent import Agent
from rl_modules.prioritised_replay_buffer import PrioritizedReplayBuffer

constants = parse_args()

class Workspace():
    def __init__(self):
        self.run_id = datetime.now().strftime('%d%H-%M%S')
        self.experiment_dir = f"{constants.experiment_dir}/{constants.experiment_name}-{self.run_id}"
        self.device = torch.device(constants.device)
        self.env = make_env(spawn_rate=constants.spawn_rate, num_knights=constants.num_knights, num_archers=constants.num_archers, killable_knights=constants.killable_knights, killable_archers=constants.killable_archers, line_death=constants.line_death)
        self.eval_env = make_env(spawn_rate=constants.spawn_rate, num_knights=constants.num_knights, num_archers=constants.num_archers, killable_knights=constants.killable_knights, killable_archers=constants.killable_archers, line_death=constants.line_death)
        self.env.reset()
        self.eval_env.reset()
        self.agent_list = self.env.agents
        self.num_of_agents = constants.num_knights + constants.num_archers
        self.action_space = self.env.action_space(self.agent_list[0])
        self.observation_space = self.env.observation_space(self.agent_list[0])
        self.total_timesteps = constants.total_timesteps
        self.timesteps = 0
        self.best_mean_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.reward_multiplier = constants.reward_multiplier
        self.agent_list = self.env.agents
        
        self.replay_buffer_size = constants.replay_buffer_size
        self.lr = constants.lr
        self.replay_start_size = constants.replay_start_size
        self.eps_decay = constants.eps_decay
        self.discount = constants.discount
        self.eps_min = constants.eps_min
        self.eps_start = constants.eps_start
        self.sync_target_network_freq = constants.sync_target_network_freq
        self.batch_size = constants.batch_size
        self.network_update_freq = constants.network_update_freq
        self.loss_function = nn.SmoothL1Loss(reduction="none")
        self.priority_rb_alpha = constants.priority_rb_alpha
        self.priority_rb_beta = constants.priority_rb_beta
        self.priority_rb_prior = constants.priority_rb_prior
        self.Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

        self.optimizers = []
        self.agents = []
        self.replay_buffers = []
        self.nets = []
        self.target_nets = []
        self.epsilons = []
        self.rewards = []

        self.prepare_experiment_dir()
        
    def prepare_experiment_dir(self):
        directory = MAKETREEDIR()
        directory.makedir(self.experiment_dir)

    def test(self):
        print(f"Action Space: {self.action_space}")
        print(f"Observation Space: {self.observation_space}")

    def evaluate(self,agent_index):
        mean_reward = np.mean(self.rewards[-100*self.num_of_agents:])
        wandb.log({"mean_reward":mean_reward})
        print(f"{self.timesteps}=> episodes:{len(self.rewards)} , mean_reward:{mean_reward} , epsilon:{self.epsilons[agent_index]}")
        
        model_weights = []

        for index in range(self.num_of_agents):
            model_weights.append(self.nets[index].state_dict())

        torch.save(model_weights, f"{self.experiment_dir}/latest-model.dat")

        if self.best_mean_reward is None or self.best_mean_reward < mean_reward:
            torch.save(self.nets[agent_index].state_dict(), f"{self.experiment_dir}/best-model.dat")
            self.best_mean_reward = mean_reward
            if self.best_mean_reward is not None:
                print("Best mean reward updated %.3f" % (self.best_mean_reward))

    def network_update(self, agent_index):
        batch = self.replay_buffers[agent_index].sample_batch(self.priority_rb_beta)
        
        # states, next_states, actions, rewards, dones,  = batch
        states = batch['obs']
        next_states = batch['next_obs']
        actions = batch['acts']
        rewards = batch['rews']
        dones = batch['done']
        weights = batch['weights']
        indices = batch['indices']

        weights = torch.FloatTensor( batch["weights"].reshape(-1, 1)).to(self.device)
        indices = batch["indices"]

        weights_v = torch.tensor(weights).to(self.device)
        states_v = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)

        # import pdb; pdb.set_trace()

        state_action_values = self.nets[agent_index](states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_nets[agent_index](next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = rewards_v + self.discount*next_state_values

        loss_t_per_element = self.loss_function(state_action_values, expected_state_action_values)
        loss = torch.mean(loss_t_per_element * weights_v)
        loss_for_prior = loss_t_per_element.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.priority_rb_prior

        wandb.log({f"loss_{agent_index}":loss})
        wandb.log({f"epsilon_{agent_index}":self.epsilons[agent_index]})
        wandb.log({f"episode_count":self.episode_count})

        self.optimizers[agent_index].zero_grad()
        loss.backward()
        self.optimizers[agent_index].step()

        if self.timesteps % self.sync_target_network_freq == 0:
            self.target_nets[agent_index].load_state_dict(self.nets[agent_index].state_dict())

        self.epsilons[agent_index] = max(self.epsilons[agent_index] - self.epsilons[agent_index]*self.eps_decay, self.eps_min)
        self.replay_buffers[agent_index].update_priorities(indices, new_priorities)
    
    def add_to_replay_buffer(self, agent_index, curr_state, action, reward, is_done, new_state):
        exp = [curr_state, action, reward, new_state, is_done]
        self.replay_buffers[agent_index].store(*exp)

    def train(self):
        for index in range(self.num_of_agents):
            self.nets.append(DQN(self.observation_space.shape,self.action_space.n).to(self.device))
            self.target_nets.append(DQN(self.observation_space.shape,self.action_space.n).to(self.device))
            self.replay_buffers.append(PrioritizedReplayBuffer( self.observation_space.shape, self.replay_buffer_size, self.batch_size, self.priority_rb_alpha ))
            self.agents.append(Agent(self.env, self.replay_buffers[index],self.action_space.n, self.device))
            self.optimizers.append(optim.Adam(self.nets[index].parameters(), lr=self.lr))
            self.epsilons.append(self.eps_start)
    
        for index_lin , agent in enumerate(self.env.agent_iter()):
            agent_index = index_lin%self.num_of_agents

            if agent_index == 0:
                self.timesteps += 1
            
            curr_state , _ , _ , _ = self.env.last()
            action = self.agents[agent_index].take_step(agent_index, self.nets[agent_index], curr_state, self.epsilons, explore_on=True)
            new_state, reward, is_done, info = self.env.last()
            reward = reward*self.reward_multiplier
            self.episode_reward += reward
            self.add_to_replay_buffer(agent_index, curr_state, action, reward, is_done, new_state)

            if is_done:
                self.rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.evaluate(agent_index)
                self.episode_count += 1
                self.env.reset()

            if (self.replay_buffers[agent_index].__len__() >= self.replay_start_size) and (self.timesteps % self.network_update_freq == 0):
                 self.network_update(agent_index)

            if self.timesteps >= self.total_timesteps:
                break


if __name__ == '__main__':
    workspace = Workspace()
    wandb.init(project=f"{constants.project_name}", name=f"{constants.experiment_name}-{workspace.run_id}_archers={constants.num_archers}_knights={constants.num_knights}_lr={constants.lr}_batch-size={constants.batch_size}_network-update-freq={constants.network_update_freq}_sync-target-network-freq={constants.sync_target_network_freq}_reward-multiplier={constants.reward_multiplier}_eps_decay={constants.eps_decay}_spawn_rate={constants.spawn_rate}")
    
    if constants.run_training_flag:
        workspace.train()
        


