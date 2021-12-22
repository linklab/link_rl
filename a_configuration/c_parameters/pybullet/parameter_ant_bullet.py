from a_configuration.b_base.a_environments.pybullet.gym_pybullet import ParameterAntBullet
from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn, ParameterDdpg
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.parameter_base import ParameterBase
from g_utils.types import ModelType


class ParameterAntBulletA2c(
    ParameterBase, ParameterAntBullet, ParameterA2c
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterAntBullet.__init__(self)
        ParameterA2c.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 1_000_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024
        self.LEARNING_RATE = 0.00001
        self.BUFFER_CAPACITY = self.BATCH_SIZE
        self.MODEL = ParameterLinearModel(ModelType.MEDIUM_LINEAR)


class ParameterAntBulletDdpg(
    ParameterBase, ParameterAntBullet, ParameterDdpg
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterAntBullet.__init__(self)
        ParameterDdpg.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 1_024
        self.LEARNING_RATE = 0.00001
        self.MODEL = ParameterLinearModel(ModelType.MEDIUM_LINEAR)

# OnPolicy