from a_configuration.base.a_environments.pybullet.gym_pybullet import ParameterAntBullet
from a_configuration.base.b_agents.agents_off_policy import ParameterDqn, ParameterDdpg
from a_configuration.base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.base.c_models.linear_layers import ParameterMediumLinearLayer
from a_configuration.base.parameter_base import ParameterBase


class ParameterAntBulletA2c(
    ParameterBase, ParameterAntBullet, ParameterA2c, ParameterMediumLinearLayer
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterAntBullet.__init__(self)
        ParameterA2c.__init__(self)
        ParameterMediumLinearLayer.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024
        self.BUFFER_CAPACITY = self.BATCH_SIZE


class ParameterAntBulletDdpg(
    ParameterBase, ParameterAntBullet, ParameterDdpg, ParameterMediumLinearLayer
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterAntBullet.__init__(self)
        ParameterDdpg.__init__(self)
        ParameterMediumLinearLayer.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024


# OnPolicy