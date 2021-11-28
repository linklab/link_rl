import numpy as np
import os
import time
import gym
import torch
import torch.multiprocessing as mp
import wandb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from a_configuration.parameter import Parameter as params
from d_agents.off_policy.dqn.agent_dqn import AgentDqn
from d_agents.on_policy.a2c.agent_a2c import AgentA2c
from d_agents.on_policy.reinforce.agent_reinforce import AgentReinforce
from e_main.supports.actor import Actor
from e_main.supports.learner import Learner
from g_utils.commons import AgentType, wandb_log, print_basic_info, get_agent
from g_utils.types import OnPolicyAgentTypes, OffPolicyAgentTypes

test_env = gym.make(params.ENV_NAME)
n_features = test_env.observation_space.shape[0]
n_actions = test_env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = get_agent(n_features, n_actions, device, params)

# 이름을 한번 언급
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
AgentType,
wandb_log
OnPolicyAgentTypes
OffPolicyAgentTypes
print_basic_info
