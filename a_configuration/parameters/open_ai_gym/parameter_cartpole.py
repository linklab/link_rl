from a_configuration.base.b_agents.agents_off_policy import ParameterDqn
from a_configuration.base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.base.c_models.linear_layers import ParameterSmallLinearLayer
from a_configuration.base.parameter_base import ParameterBase
from a_configuration.base.a_environments.open_ai_gym.gym_classic_control import ParameterCartPole


class ParameterCartPoleDqn(
    ParameterBase, ParameterCartPole, ParameterDqn, ParameterSmallLinearLayer
):
    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    MAX_TRAINING_STEPS = 100_000
    CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = 200


# OnPolicy

class ParameterCartPoleReinforce(
    ParameterBase, ParameterCartPole, ParameterReinforce, ParameterSmallLinearLayer
):
    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    BUFFER_CAPACITY = 1_000
    MAX_TRAINING_STEPS = 10_000
    CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = 200


class ParameterCartPoleA2c(
    ParameterBase, ParameterCartPole, ParameterA2c, ParameterSmallLinearLayer
):
    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    MAX_TRAINING_STEPS = 100_000
    BUFFER_CAPACITY = 100_000
    BATCH_SIZE = 32
    CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = 200
