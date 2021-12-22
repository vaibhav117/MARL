## Multi-Agent Reinforcement Learning (MARL) to master Knights Archers Zombies game

### Goal
In this project, our aim is to implement a Cooperative MARL model with the goal of mastering the KAZ game that involves multiple agents.

### Main files

- arguments.py: Stores all the variables and values of the hyper-parameter.
- env.py: Imports, initializes and pre-processes KAZ environment from PettingZoo library.
- agent.py: Selects actions according to the current state and epsilon value, and takes steps for each agent individually
- dqn.py: The DQN MARL algorithm
- replay_buffer.py: Stores the trajectories in the memory
- train.py: Trains the agents, updates replay buffer, calculates loss and updates the network.
- test_pettingzoo.py: Uses the optimum model to test on a fresh environment.
- video.py: Captures the frame, records the progress, saves the video and allows us to watch them play

### Instructions to run
1. 


### Demo
Single agent

![](Single_agent.gif)

Multiple agents

![](Multi_agents.gif)

---
### NYU Deep Learning Final Project
### Harini Appansrinivasan (ha1642) and Vaibhav Mathur (vm2134)
