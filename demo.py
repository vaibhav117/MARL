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
        self.run_id = constants.run_id
        self.experiment_dir = f"{constants.experiment_dir}/{constants.experiment_name}-{self.run_id}"
        self.device = torch.device(constants.device)
        self.env = make_env(spawn_rate=constants.spawn_rate, num_knights=constants.num_knights, num_archers=constants.num_archers, killable_knights=constants.killable_knights, killable_archers=constants.killable_archers, line_death=constants.line_death)
        self.eval_env = make_env(spawn_rate=5, num_knights=constants.num_knights, num_archers=constants.num_archers, killable_knights=constants.killable_knights, killable_archers=constants.killable_archers, line_death=constants.line_death)
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

    def init_demo(self):
        for index in range(self.num_of_agents):
            self.nets.append(DQN(self.observation_space.shape,self.action_space.n).to(self.device))
            self.replay_buffers.append(ReplayBuffer( self.replay_buffer_size ))
            self.agents.append(Agent(self.env, self.replay_buffers[index],self.action_space.n, self.device))
            self.epsilons.append(0.2)
        
    def load_model(self, model_path):
        for agent_index in range(self.num_of_agents):
            # import pdb; pdb.set_trace()
            model = torch.load(model_path, map_location=lambda storage, loc: storage)[agent_index]
            self.nets[agent_index].load_state_dict(model)
            self.nets[agent_index].eval()

    def play(self):
        for index_lin , agent in enumerate(self.env.agent_iter()):
            agent_index = index_lin % self.num_of_agents
            obs, reward, done, info = self.env.last()
            action = self.agents[agent_index].take_step(agent_index, self.nets[agent_index], obs, self.epsilons, explore_on=True)
            
            if agent == 'archer_0':
                print(action) 
            # plt.imshow(obs)
            # plt.show()

            self.env.render(mode='human')
            time.sleep(0.1)

if __name__ == '__main__':    
    if constants.run_id == None:
        print("\n---------- Please pass a proper run_id for creating the demo ----------\n")
    
    workspace = Workspace()
    workspace.init_demo()
    workspace.load_model("trained_models/new_4_agent_latest.dat")
    workspace.play()
