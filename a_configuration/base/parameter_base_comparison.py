import os
import sys

from a_configuration.parameters.open_ai_gym.parameter_cartpole_a2c import ParameterCartPoleA2c
from a_configuration.parameters.open_ai_gym.parameter_cartpole_dqn import ParameterCartPoleDqn
from g_utils.commons import AgentType


class ParameterBaseComparison:
    PROJECT_HOME = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
    )
    if PROJECT_HOME not in sys.path:
        sys.path.append(PROJECT_HOME)

    AGENTS = [
        AgentType.Dqn,
        AgentType.A2c
    ]
    PARAMS_AGENTS = [
        ParameterCartPoleDqn,
        ParameterCartPoleA2c
    ]

    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    MAX_TRAINING_STEPS = 100_000

    TRAIN_INTERVAL_TOTAL_TIME_STEPS = 4
    assert TRAIN_INTERVAL_TOTAL_TIME_STEPS >= N_VECTORIZED_ENVS * N_ACTORS, \
        "TRAIN_INTERVAL_TOTAL_TIME_STEPS should be greater than N_VECTORIZED_ENVS * N_ACTORS"

    CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = 100
    TEST_INTERVAL_TOTAL_TIME_STEPS = 1_000

    N_EPISODES_FOR_MEAN_CALCULATION = 100

    N_TEST_EPISODES = 3

    USE_WANDB = False
    WANDB_ENTITY = "link-koreatech"

