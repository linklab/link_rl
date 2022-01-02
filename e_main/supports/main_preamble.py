import numpy as np
import os
from gym.spaces import Discrete, Box
import warnings

from d_agents.off_policy.ddpg.agent_ddpg import AgentDdpg
from d_agents.off_policy.dqn.agent_double_dqn import AgentDoubleDqn
from d_agents.off_policy.dqn.agent_dueling_dqn import AgentDuelingDqn
from d_agents.off_policy.sac.agent_sac import AgentSac

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from d_agents.off_policy.dqn.agent_dqn import AgentDqn
from d_agents.on_policy.a2c.agent_a2c import AgentA2c
from d_agents.on_policy.reinforce.agent_reinforce import AgentReinforce
from g_utils.commons import AgentType

from gym import logger
logger.set_level(level=40)


def get_agent(observation_space, action_space, parameter=None):
    assert isinstance(observation_space, Box)

    if parameter.AGENT_TYPE == AgentType.DQN:
        assert isinstance(action_space, Discrete)
        agent = AgentDqn(
            observation_space=observation_space, action_space=action_space, parameter=parameter
        )
    elif parameter.AGENT_TYPE == AgentType.DOUBLE_DQN:
        agent = AgentDoubleDqn(
            observation_space=observation_space, action_space=action_space, parameter=parameter
        )
    elif parameter.AGENT_TYPE == AgentType.DUELING_DQN:
        agent = AgentDuelingDqn(
            observation_space=observation_space, action_space=action_space, parameter=parameter
        )
    elif parameter.AGENT_TYPE == AgentType.REINFORCE:
        assert parameter.N_ACTORS * parameter.N_VECTORIZED_ENVS == 1, "TOTAL NUMBERS OF ENVS should be one"
        agent = AgentReinforce(
            observation_space=observation_space, action_space=action_space, parameter=parameter
        )
    elif parameter.AGENT_TYPE == AgentType.A2C:
        agent = AgentA2c(
            observation_space=observation_space, action_space=action_space, parameter=parameter
        )
    elif parameter.AGENT_TYPE == AgentType.DDPG:
        agent = AgentDdpg(
            observation_space=observation_space, action_space=action_space, parameter=parameter
        )
    elif parameter.AGENT_TYPE == AgentType.SAC:
        agent = AgentSac(
            observation_space=observation_space, action_space=action_space, parameter=parameter
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