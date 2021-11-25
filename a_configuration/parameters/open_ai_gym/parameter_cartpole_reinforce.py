from a_configuration.base.agents.parameter_agents_on_policy import ParameterReinforce
from a_configuration.base.parameter_base import ParameterBase
from a_configuration.base.environments.open_ai_gym.parameter_gym_classic_control import ParameterCartPole


class ParameterCartPoleReinforce(
    ParameterBase, ParameterCartPole, ParameterReinforce
):
    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    BUFFER_CAPACITY = 1_000
    MAX_TRAINING_STEPS = 10_000
