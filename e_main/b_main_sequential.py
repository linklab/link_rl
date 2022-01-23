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
from g_utils.commons_rl import set_parameters, get_agent
from e_main.parameter import parameter

from a_configuration.a_config.config import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
parameter.SYSTEM_USER_NAME = SYSTEM_USER_NAME
parameter.SYSTEM_COMPUTER_NAME = SYSTEM_COMPUTER_NAME


def main():
    set_parameters(parameter)

    observation_space, action_space = get_env_info(parameter)
    print_basic_info(observation_space, action_space, parameter)

    input("Press Enter to continue...")

    agent = get_agent(
        observation_space=observation_space, action_space=action_space, parameter=parameter
    )

    learner = Learner(agent=agent, queue=None, parameter=parameter)

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop(parallel=False)

    print_basic_info(observation_space, action_space, parameter)


if __name__ == "__main__":
    assert parameter.N_ACTORS == 1
    main()
