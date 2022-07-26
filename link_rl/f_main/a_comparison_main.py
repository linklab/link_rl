import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from gym import logger
logger.set_level(level=40)

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from link_rl.h_utils.types import AgentType
from link_rl.h_utils.stats import ComparisonStat
from link_rl.f_main.config_comparison import config_c

from link_rl.a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
config_c.SYSTEM_USER_NAME = SYSTEM_USER_NAME
config_c.SYSTEM_COMPUTER_NAME = SYSTEM_COMPUTER_NAME

from link_rl.f_main.supports.learner_comparison import LearnerComparison
from link_rl.h_utils.commons import print_comparison_basic_info, get_wandb_obj, get_env_info
from link_rl.h_utils.commons import set_config
from link_rl.h_utils.commons_rl import get_agent

n_agents = len(config_c.AGENT_PARAMETERS)

import random
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

if config_c.SEED is not None:
    random.seed(config_c.SEED)
    torch.manual_seed(config_c.SEED)
    np.random.seed(config_c.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for agent_config in config_c.AGENT_PARAMETERS:
    # del agent_config.MAX_TRAINING_STEPS
    del agent_config.N_ACTORS
    del agent_config.N_TEST_EPISODES
    # del agent_config.N_VECTORIZED_ENVS
    del agent_config.PROJECT_HOME
    del agent_config.TEST_INTERVAL_TRAINING_STEPS
    del agent_config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
    del agent_config.USE_WANDB
    del agent_config.MODEL_SAVE_DIR
    del agent_config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS


def main():
    for config in config_c.AGENT_PARAMETERS:
        assert config.AGENT_TYPE not in (AgentType.REINFORCE,)
        set_config(config)

    observation_space, action_space = get_env_info(config_c.AGENT_PARAMETERS[0])
    print_comparison_basic_info(observation_space, action_space, config_c)

    input("Press Enter (two or more times) to continue...")

    if config_c.USE_WANDB:
        wandb_obj = get_wandb_obj(config_c, comparison=True)
    else:
        wandb_obj = None

    comparison_stat = ComparisonStat(config_c=config_c)

    print("\n########## LEARNING STARTED !!! ##########")
    for run in range(0, config_c.N_RUNS):
        print("\n" + ">" * 30 + " RUN: {0} ".format(run + 1) + "<" * 30)
        agents = []
        for config in config_c.AGENT_PARAMETERS:
            assert config.AGENT_TYPE not in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]
            observation_space, action_space = get_env_info(config)
            agent = get_agent(
                observation_space=observation_space, action_space=action_space,
                config=config
            )
            agents.append(agent)

        learner_comparison = LearnerComparison(
            run=run, agents=agents, wandb_obj=wandb_obj, config_c=config_c, comparison_stat=comparison_stat
        )
        learner_comparison.train_comparison_loop(run=run)

    if config_c.USE_WANDB:
        wandb_obj.finish()

    print_comparison_basic_info(observation_space, action_space, config_c)

    now = datetime.datetime.now()
    local_now = now.astimezone()
    comparison_stat.save_csv(local_now)
    comparison_stat.save_fig(local_now)


if __name__ == "__main__":
    # assert config.AGENT_TYPE in OnPolicyAgentTypes
    assert config_c.N_ACTORS == 1 and config_c.N_VECTORIZED_ENVS == 1
    main()
