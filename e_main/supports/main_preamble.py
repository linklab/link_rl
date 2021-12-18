import numpy as np
import os
from gym.spaces import Discrete, Box
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("No Warning Shown")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import gym
import torch
import torch.multiprocessing as mp
import wandb

import sys

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from a_configuration.parameter import Parameter
from d_agents.off_policy.dqn.agent_dqn import AgentDqn
from d_agents.on_policy.a2c.agent_a2c import AgentA2c
from d_agents.on_policy.reinforce.agent_reinforce import AgentReinforce
from e_main.supports.actor import Actor
from e_main.supports.learner import Learner
from g_utils.commons import AgentType, wandb_log, print_basic_info
from g_utils.types import OnPolicyAgentTypes, OffPolicyAgentTypes
from g_utils.commons import get_env_info


def get_agent(observation_space, action_space, device=torch.device("cpu"), parameter=None, max_training_steps=None):
    assert isinstance(observation_space, Box)
    observation_shape = observation_space.shape

    if parameter.AGENT_TYPE == AgentType.Dqn:
        assert isinstance(action_space, Discrete)
        agent = AgentDqn(
            observation_space=observation_space, action_space=action_space, device=device, parameter=parameter,
            max_training_steps=max_training_steps
        )
    elif parameter.AGENT_TYPE == AgentType.Reinforce:
        assert parameter.N_ACTORS * parameter.N_VECTORIZED_ENVS == 1, \
            "TOTAL NUMBERS OF ENVS should be one"

        agent = AgentReinforce(
            observation_space=observation_space, action_space=action_space, device=device, parameter=parameter
        )
    elif parameter.AGENT_TYPE == AgentType.A2c:
        agent = AgentA2c(
            observation_space=observation_space, action_space=action_space, device=device, parameter=parameter
        )
    else:
        raise ValueError()

    return agent


# 이름을 한번 언급
# time
# gym
# torch
# mp
# wandb
# AgentDqn
# AgentA2c
# AgentReinforce
# Actor
# Learner
# AgentType,
# wandb_log
# OnPolicyAgentTypes
# OffPolicyAgentTypes
# print_basic_info
# get_env_info
# Parameter