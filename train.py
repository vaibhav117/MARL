from rl_modules.dqn.dqn import DQN
import torch
import torch.optim as optim
import torch.nn as nn 

import numpy as np
import time
import wandb

from env import make_env
from arguments import parse_args
from rl_modules.dqn.agent import Agent
from rl_modules.dqn.replay_buffer import ReplayBuffer

constants = parse_args()

class Workspace():
    def __init__(self):
        self.device = torch.device(constants.device)
        self.env = make_env(num_knights=constants.num_knights,num_archers=constants.num_archers,killable_knights=constants.killable_knights,killable_archers=constants.killable_archers,line_death=constants.line_death)
        self.eval_env = make_env(num_knights=constants.num_knights,num_archers=constants.num_archers,killable_knights=constants.killable_knights,killable_archers=constants.killable_archers,line_death=constants.line_death)
        self.env.reset()
        self.eval_env.reset()
        self.agent_list = self.env.agents
        self.num_of_agents = constants.num_knights + constants.num_archers
        self.action_space = self.env.action_space(self.agent_list[0])
        self.observation_space = self.env.observation_space(self.agent_list[0])
        self.total_timesteps = constants.total_timesteps
        self.timesteps = 0
        self.best_mean_reward = 0
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
        self.episode_count = 0

        self.optimizers = []
        self.agents = []
        self.replay_buffers = []
        self.nets = []
        self.target_nets = []
        self.epsilons = []
        self.rewards = []
        

    def test(self):
        print(f"Action Space: {self.action_space}")
        print(f"Observation Space: {self.observation_space}")

    def evaluate(self,agent_index):
        mean_reward = np.mean(self.rewards[-100*self.num_of_agents:])
        wandb.log({"mean_reward":mean_reward})
        print(f"{self.timesteps}=> episodes:{len(self.rewards)} , mean_reward:{mean_reward} , epsilon:{self.epsilons[agent_index]}")
        if self.best_mean_reward is None or self.best_mean_reward < mean_reward:
            torch.save(self.nets[agent_index].state_dict(), "best-model.dat")
            self.best_mean_reward = mean_reward
            if self.best_mean_reward is not None:
                print("Best mean reward updated %.3f" % (self.best_mean_reward))

    def network_update(self,agent_index):
        print(f"----- Performing network update -----")
        batch = self.replay_buffers[agent_index].sample(self.batch_size)
        
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)

        state_action_values = self.nets[agent_index](states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_nets[agent_index](next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.discount + rewards_v

        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)
        wandb.log({"loss":loss_t})

        self.optimizers[agent_index].zero_grad()
        loss_t.backward()
        self.optimizers[agent_index].step()

        if self.timesteps % self.sync_target_network_freq == 0:
            self.target_nets[agent_index].load_state_dict(self.nets[agent_index].state_dict())

        self.epsilons[agent_index] = max(self.epsilons[agent_index]*self.eps_decay, self.eps_min)


    def train(self):
        for index in range(self.num_of_agents):
            self.nets.append(DQN(self.observation_space.shape,self.action_space.n).to(self.device))
            self.target_nets.append(DQN(self.observation_space.shape,self.action_space.n).to(self.device))
            self.replay_buffers.append(ReplayBuffer( self.replay_buffer_size ))
            self.agents.append(Agent(self.env, self.replay_buffers[index],self.action_space.n))
            self.optimizers.append(optim.Adam(self.nets[index].parameters(), lr=self.lr))
            self.epsilons.append(self.eps_start)
        

        for index_lin , agent in enumerate(self.env.agent_iter()):
            self.timesteps += 1
            index = index_lin%self.num_of_agents
            reward = self.agents[index].play_step(self.nets[index], self.epsilons[index], device=self.device)
            if reward is not None:
                self.rewards.append(reward)
                self.evaluate(index)
                self.episode_count += 1

            if (self.replay_buffers[index].__len__() >= self.replay_start_size) and (self.timesteps % self.network_update_freq):
                 self.network_update(index)

            if self.timesteps >= self.total_timesteps:
                break


if __name__ == '__main__':
    workspace = Workspace()
    wandb.init(project=f"{constants.project_name}", name=f"{constants.experiment_name}_archers={constants.num_archers}_knights={constants.num_knights}_lr={constants.lr}_batch-size={constants.batch_size}")
    workspace.train()
    if constants.run_training_flag:
        pass
        


