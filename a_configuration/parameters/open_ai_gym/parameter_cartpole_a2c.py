from a_configuration.base.agents.parameter_agents_on_policy import ParameterA2c
from a_configuration.base.parameter_base import ParameterBase
from a_configuration.base.environments.open_ai_gym.parameter_gym_classic_control import ParameterCartPole


class ParameterCartPoleA2c(
    ParameterBase, ParameterCartPole, ParameterA2c
):
    N_VECTORIZED_ENVS = 2
    N_ACTORS = 2
    MAX_TRAINING_STEPS = 100_000
    BUFFER_CAPACITY = 100_000
