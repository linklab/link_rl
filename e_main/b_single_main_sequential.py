import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

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

from e_main.supports.learner import Learner
from g_utils.commons import get_env_info, print_basic_info
from g_utils.commons import set_config
from g_utils.commons_rl import get_agent
from e_main.config_single import config

from a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
config.SYSTEM_USER_NAME = SYSTEM_USER_NAME
config.SYSTEM_COMPUTER_NAME = SYSTEM_COMPUTER_NAME


def main():
    set_config(config)

    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)

    input("Press Enter to continue...")

    agent = get_agent(
        observation_space=observation_space, action_space=action_space, config=config
    )

    learner = Learner(agent=agent, queue=None, config=config)

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop(parallel=False)

    print_basic_info(observation_space, action_space, config)


if __name__ == "__main__":
    assert config.N_ACTORS == 1, "Current config.N_ACTORS: {0}".format(config.N_ACTORS)
    main()
