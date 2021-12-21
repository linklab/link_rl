from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.b_base.c_models.linear_layers import ParameterSmallLinearLayer
from a_configuration.b_base.parameter_base import ParameterBase
from a_configuration.b_base.a_environments.open_ai_gym.gym_classic_control import ParameterCartPole


class ParameterCartPoleDqn(
    ParameterBase, ParameterCartPole, ParameterDqn, ParameterSmallLinearLayer
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterDqn.__init__(self)
        ParameterSmallLinearLayer.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100

# OnPolicy

class ParameterCartPoleReinforce(
    ParameterBase, ParameterCartPole, ParameterReinforce, ParameterSmallLinearLayer
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterReinforce.__init__(self)
        ParameterSmallLinearLayer.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.BUFFER_CAPACITY = 1_000
        self.MAX_TRAINING_STEPS = 10_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100


class ParameterCartPoleA2c(
    ParameterBase, ParameterCartPole, ParameterA2c, ParameterSmallLinearLayer
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterA2c.__init__(self)
        ParameterSmallLinearLayer.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 100_000
        self.BATCH_SIZE = 32
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
