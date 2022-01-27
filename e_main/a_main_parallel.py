import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import time

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from gym import logger
logger.set_level(level=40)

import torch.multiprocessing as mp
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.config import config
from e_main.supports.actor import Actor
from e_main.supports.learner import Learner
from g_utils.commons import get_env_info, print_basic_info
from g_utils.types import OffPolicyAgentTypes
from g_utils.commons_rl import set_config, get_agent

from a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME, SYSTEM_COMPUTER_NAME
config.SYSTEM_USER_NAME = SYSTEM_USER_NAME
config.SYSTEM_COMPUTER_NAME = SYSTEM_COMPUTER_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_config(config)
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)

    input("Press Enter to continue...")

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()

    agent = get_agent(
        observation_space=observation_space, action_space=action_space, config=config
    )

    learner = Learner(agent=agent, queue=queue, config=config)

    actors = [
        Actor(
            env_name=config.ENV_NAME, actor_id=actor_id, agent=agent, queue=queue, config=config
        ) for actor_id in range(config.N_ACTORS)
    ]

    for actor in actors:
        actor.start()

    # Busy Wait: 모든 액터들이 VecEnv를 생성 완료할 때까지 대기
    for actor in actors:
        while not actor.is_vectorized_env_created.value:
            time.sleep(0.1)

    print("########## LEARNING STARTED !!! ##########")

    learner.start()

    while True:
        # Busy Wait: learner에서 학습 완료될 때까지 대기
        if learner.is_terminated.value:
            # learner가 학습 완료하면 각 actor들의 rollout 종료
            for actor in actors:
                actor.is_terminated.value = True
            break
        time.sleep(0.5)

    # Busy Wait: 모든 actor가 조인할 때까지 대기
    while any([actor.is_alive() for actor in actors]):
        for actor in actors:
            actor.join(timeout=1)

    # Busy Wait: learner가 조인할 때까지 대기
    while learner.is_alive():
        learner.join(timeout=1)

    print_basic_info(observation_space, action_space, config)


if __name__ == "__main__":
    assert config.AGENT_TYPE in OffPolicyAgentTypes
    main()
