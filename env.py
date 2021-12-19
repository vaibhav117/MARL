from pettingzoo.butterfly import knights_archers_zombies_v7
import warnings
warnings.filterwarnings("ignore")
from pettingzoo.utils import random_demo
import supersuit as ss
import numpy as np
import time
import gym

def make_env(spawn_rate=20, num_knights=0, num_archers=1, killable_knights=False, killable_archers=False, line_death=False):
    env = knights_archers_zombies_v7.env(num_knights=num_knights , num_archers=num_archers, killable_knights=killable_knights, killable_archers=killable_archers, line_death=line_death)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v0(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    env = ss.dtype_v0(env, dtype="float32")
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ImageToPyTorch(env)
    # env = ss.pettingzoo_env_to_vec_env_v0(env)
    # env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=4, base_class='stable_baselines3')
    return env
