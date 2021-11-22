from pettingzoo.butterfly import knights_archers_zombies_v7
import warnings
warnings.filterwarnings("ignore")
from pettingzoo.utils import random_demo
import supersuit as ss
import numpy as np
import time

# class ImageToPyTorch(gym.ObservationWrapper):
#     def __init__(self, env):
#         super(ImageToPyTorch, self).__init__(env)
#         old_shape = self.observation_space.shape
#         self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], 
#                                 old_shape[0], old_shape[1]), dtype=np.float32)

#     def observation(self, observation):
#         return np.moveaxis(observation, 2, 0)

def make_env(num_knights=0,num_archers=1):
    env = knights_archers_zombies_v7.env(num_knights=num_knights , num_archers=num_archers)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ImageToPyTorch(env)
    # env = ss.pettingzoo_env_to_vec_env_v0(env)
    # env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=4, base_class='stable_baselines3')
    return env
