from pettingzoo.butterfly import knights_archers_zombies_v7
import warnings
warnings.filterwarnings("ignore")
from pettingzoo.utils import random_demo
import numpy as np
import time



def make_env(num_knights=0,num_archers=1):
    env = knights_archers_zombies_v7.env(num_knights=num_knights , num_archers=num_archers)
    return env
