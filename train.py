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
from rl_modules.dqn.replay_buffer import ReplayBuffer

constants = parse_args()

"""
This is the main training file. Here
"""

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
        self.loss_function = nn.SmoothL1Loss()
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

    def evaluate(self,agent_index):
        mean_reward = np.mean(self.rewards[-100*self.num_of_agents:])
        wandb.log({"mean_reward":mean_reward})
        print(f"{self.timesteps}=> episodes:{len(self.rewards)} , mean_reward:{mean_reward} , epsilon:{self.epsilons[agent_index]}")
        
        model_weights = []

        for index in range(self.num_of_agents):
            model_weights.append(self.nets[index].state_dict())

        torch.save(model_weights, f"{self.experiment_dir}/latest-model.dat")

        # Updating saved models if the mean_reward is higher than the best
        if self.best_mean_reward is None or self.best_mean_reward < mean_reward:
            torch.save(model_weights, f"{self.experiment_dir}/best-model.dat")
            self.best_mean_reward = mean_reward
            if self.best_mean_reward is not None:
                print("Best mean reward updated %.3f" % (self.best_mean_reward))

    def network_update(self, agent_index):
        batch = self.replay_buffers[agent_index].sample(self.batch_size)
        
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)

        state_action_values = self.nets[agent_index](states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        # Getting the Q(s_{t+1}) value using the target network to calculate the Bellman Loss
        next_state_values = self.target_nets[agent_index](next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = rewards_v + self.discount*next_state_values

        # We're using SmoothL1Loss instead of MSELoss as it's more stable
        loss_t = self.loss_function(state_action_values, expected_state_action_values)
        wandb.log({f"loss_{agent_index}":loss_t})
        wandb.log({f"epsilon_{agent_index}":self.epsilons[agent_index]})
        wandb.log({f"episode_count":self.episode_count})

        # Calculating gradient and running backpropagation
        self.optimizers[agent_index].zero_grad()
        loss_t.backward()
        self.optimizers[agent_index].step()

        # Updating the Target Network every 100 updates to the main network.
        # This is required to stabalize learning
        if self.timesteps % self.sync_target_network_freq == 0:
            self.target_nets[agent_index].load_state_dict(self.nets[agent_index].state_dict())

        # Decaying exploration as the models begin to learn
        self.epsilons[agent_index] = max(self.epsilons[agent_index] - self.epsilons[agent_index]*self.eps_decay, self.eps_min)

    
    def add_to_replay_buffer(self, agent_index, curr_state, action, reward, is_done, new_state):
        exp = self.Experience(curr_state, action, reward, is_done, new_state)
        self.replay_buffers[agent_index].append(exp)

    def train(self):
        for index in range(self.num_of_agents):
            self.nets.append(DQN(self.observation_space.shape,self.action_space.n).to(self.device))
            self.target_nets.append(DQN(self.observation_space.shape,self.action_space.n).to(self.device))
            self.replay_buffers.append(ReplayBuffer( self.replay_buffer_size ))
            self.agents.append(Agent(self.env, self.replay_buffers[index],self.action_space.n, self.device))
            self.optimizers.append(optim.Adam(self.nets[index].parameters(), lr=self.lr))
            self.epsilons.append(self.eps_start)
        
        # Here we iterate over the agents using the agent_iter() function provided by petting zoo
        for index_lin , agent in enumerate(self.env.agent_iter()):
            agent_index = index_lin%self.num_of_agents

            # Timestep is updated only on a new cycle of agent updates
            # as the environemnt is syncronously moves forward in a time 
            # frame ony after all agents of decided their steps for the 
            # current state
            if agent_index == 0:
                self.timesteps += 1
            
            # Petting returns agent observations, reward and other info using env.last() function
            # for the agent currently in play
            curr_state, _, _, _ = self.env.last()

            # Here we call the agent to use the policy and take the step in the environment and return the action
            action = self.agents[agent_index].take_step(agent_index, self.nets[agent_index], curr_state, self.epsilons, explore_on=True)
            new_state, reward, is_done, info = self.env.last()

            # We are using an reward multiplier of value 10 for every kill to boost learning
            reward = reward*self.reward_multiplier                  
            self.episode_reward += reward

            # Adding the transition to the replay buffer
            self.add_to_replay_buffer(agent_index, curr_state, action, reward, is_done, new_state)
            
            # Env reset is done after the final agent has decided it's step
            # for the current state
            if is_done and agent_index == self.num_of_agents-1 :
                self.rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.evaluate(agent_index)
                self.episode_count += 1
                self.env.reset()

            # Network updates are called every network_update_freq timesteps
            if (self.replay_buffers[agent_index].__len__() >= self.replay_start_size) and (self.timesteps % self.network_update_freq == 0):
                 self.network_update(agent_index)

            if self.timesteps >= self.total_timesteps:
                break


if __name__ == '__main__':
    workspace = Workspace()
    wandb.init(project=f"{constants.project_name}", name=f"{constants.experiment_name}-{workspace.run_id}_archers={constants.num_archers}_knights={constants.num_knights}_lr={constants.lr}_batch-size={constants.batch_size}_network-update-freq={constants.network_update_freq}_sync-target-network-freq={constants.sync_target_network_freq}_reward-multiplier={constants.reward_multiplier}_eps_decay={constants.eps_decay}_spawn_rate={constants.spawn_rate}")
    
    if constants.run_training_flag:
        workspace.train()
        


