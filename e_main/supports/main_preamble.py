import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

import time
import gym
import torch
import torch.multiprocessing as mp
import wandb

from d_agents.off_policy.dqn.agent_dqn import AgentDqn
from d_agents.on_policy.a2c.agent_a2c import AgentA2c
from d_agents.on_policy.reinforce.agent_reinforce import AgentReinforce
from e_main.supports.actor import Actor
from e_main.supports.learner import Learner
from g_utils.commons import print_params, AgentType, wandb_log

time
gym
torch
mp
wandb
AgentDqn
AgentA2c
AgentReinforce
Actor
Learner
print_params,
AgentType,
wandb_log

mp.set_start_method('spawn', force=True)
