import warnings

from g_utils.types import AgentType

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import datetime
import torch
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

from g_utils.stats import ComparisonStat
from e_main.config_comparison import config_c

from a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
config_c.SYSTEM_USER_NAME = SYSTEM_USER_NAME
config_c.SYSTEM_COMPUTER_NAME = SYSTEM_COMPUTER_NAME

from e_main.supports.learner_comparison import LearnerComparison
from g_utils.commons import print_comparison_basic_info, get_wandb_obj, get_env_info
from g_utils.commons import set_config
from g_utils.commons_rl import get_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_agents = len(config_c.AGENT_PARAMETERS)

for agent_config in config_c.AGENT_PARAMETERS:
    #del agent_config.MAX_TRAINING_STEPS
    del agent_config.N_ACTORS
    del agent_config.N_EPISODES_FOR_MEAN_CALCULATION
    del agent_config.N_TEST_EPISODES
    del agent_config.N_VECTORIZED_ENVS
    del agent_config.PROJECT_HOME
    del agent_config.TEST_INTERVAL_TRAINING_STEPS
    del agent_config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
    del agent_config.USE_WANDB
    del agent_config.WANDB_ENTITY
    del agent_config.MODEL_SAVE_DIR
    del agent_config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS


def main():
    for config in config_c.AGENT_PARAMETERS:
        assert config.AGENT_TYPE != AgentType.REINFORCE
        assert config.AGENT_TYPE != AgentType.PPO
        set_config(config)

    observation_space, action_space = get_env_info(config_c)
    print_comparison_basic_info(observation_space, action_space, config_c)

    input("Press Enter to continue...")

    if config_c.USE_WANDB:
        wandb_obj = get_wandb_obj(config_c, comparison=True)
    else:
        wandb_obj = None

    comparison_stat = ComparisonStat(config_c=config_c)

    print("\n########## LEARNING STARTED !!! ##########")
    for run in range(0, config_c.N_RUNS):
        print("\n" + ">" * 30 + " RUN: {0} ".format(run + 1) + "<" * 30)
        agents = []
        for agent_idx, _ in enumerate(config_c.AGENT_PARAMETERS):
            agent = get_agent(
                observation_space=observation_space, action_space=action_space,
                config=config_c.AGENT_PARAMETERS[agent_idx]
            )
            agents.append(agent)

        learner_comparison = LearnerComparison(
            run=run, agents=agents, wandb_obj=wandb_obj, config_c=config_c, comparison_stat=comparison_stat
        )
        learner_comparison.train_comparison_loop()

    if config_c.USE_WANDB:
        wandb_obj.join()

    print_comparison_basic_info(observation_space, action_space, config_c)

    now = datetime.datetime.now()
    local_now = now.astimezone()
    comparison_stat.save_csv(local_now)
    comparison_stat.save_fig(local_now)


if __name__ == "__main__":
    # assert config.AGENT_TYPE in OnPolicyAgentTypes
    assert config_c.N_ACTORS == 1 and config_c.N_VECTORIZED_ENVS == 1
    main()
