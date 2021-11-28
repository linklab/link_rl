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
from g_utils.commons import AgentType, wandb_log, print_basic_info
from g_utils.types import OnPolicyAgentTypes, OffPolicyAgentTypes
from g_utils.commons import get_test_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_agent(obs_shape, n_actions, device=torch.device("cpu"), params=None):
    if params.AGENT_TYPE == AgentType.Dqn:
        agent = AgentDqn(
            obs_shape=obs_shape, n_actions=n_actions, device=device, params=params
        )
    elif params.AGENT_TYPE == AgentType.Reinforce:
        assert params.N_ACTORS * params.N_VECTORIZED_ENVS == 1, \
            "TOTAL NUMBERS OF ENVS should be one"

        agent = AgentReinforce(
            obs_shape=obs_shape, n_actions=n_actions, device=device, params=params
        )
    elif params.AGENT_TYPE == AgentType.A2c:
        agent = AgentA2c(
            obs_shape=obs_shape, n_actions=n_actions, device=device, params=params
        )
    else:
        raise ValueError()

    return agent


def get_agents(n_features, n_actions, device, params_c):
    agents = []
    for idx, agent_type in enumerate(params_c.AGENTS):
        agent_params = params_c.PARAMS_AGENTS[idx]
        if agent_type == AgentType.Dqn:
            agents.append(
                AgentDqn(
                    n_features=n_features, n_actions=n_actions, device=device,
                    params=agent_params
                )
            )
        elif agent_type == AgentType.Reinforce:
            assert agent_params.N_ACTORS * agent_params.N_VECTORIZED_ENVS == 1, \
                "AGENT_REINFORCE: TOTAL NUMBERS OF ENVS should be one"
            agents.append(
                AgentReinforce(
                    n_features=n_features, n_actions=n_actions, device=device,
                    params=agent_params
                )
            )
        elif agent_type == AgentType.A2c:
            agents.append(
                AgentA2c(
                    n_features=n_features, n_actions=n_actions, device=device,
                    params=agent_params
                )
            )
        else:
            raise ValueError()

    return agents



# 이름을 한번 언급
params
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
get_test_env