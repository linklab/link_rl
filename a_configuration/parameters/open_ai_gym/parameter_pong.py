from a_configuration.base.agents.agents_off_policy import ParameterDqn
from a_configuration.base.environments.open_ai_gym.gym_atari import ParameterPong
from a_configuration.base.parameter_base import ParameterBase


class ParameterPongDqn(
    ParameterBase, ParameterPong, ParameterDqn
):
    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    MAX_TRAINING_STEPS = 100_000
    CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = 200
