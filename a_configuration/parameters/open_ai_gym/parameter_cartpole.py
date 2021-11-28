from a_configuration.base.agents.parameter_agents_off_policy import ParameterDqn
from a_configuration.base.agents.parameter_agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.base.parameter_base import ParameterBase
from a_configuration.base.environments.open_ai_gym.parameter_gym_classic_control import ParameterCartPole


################
## OFF_POLICY ##
################
class ParameterCartPoleDqn(
    ParameterBase, ParameterCartPole, ParameterDqn
):
    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    MAX_TRAINING_STEPS = 100_000
    CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = 200

###############
## ON_POLICY ##
###############
class ParameterCartPoleReinforce(
    ParameterBase, ParameterCartPole, ParameterReinforce
):
    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    BUFFER_CAPACITY = 1_000
    MAX_TRAINING_STEPS = 10_000
    CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = 200

class ParameterCartPoleA2c(
    ParameterBase, ParameterCartPole, ParameterA2c
):
    N_VECTORIZED_ENVS = 1
    N_ACTORS = 1
    MAX_TRAINING_STEPS = 100_000
    BUFFER_CAPACITY = 100_000
    CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS = 200
    