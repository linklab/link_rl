from a_configuration.b_base.parameter_base_comparison import ParameterComparisonBase
from a_configuration.c_parameters.open_ai_gym.parameter_cart_pole import ParameterCartPoleDqn, ParameterCartPoleReinforce, \
    ParameterCartPoleA2c
from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletA2c, \
    ParameterCartPoleBulletDqn, ParameterCartPoleBulletDoubleDqn, ParameterCartPoleBulletDuelingDqn, \
    ParameterCartPoleBulletDoubleDuelingDqn
from a_configuration.c_parameters.pybullet.parameter_cartpole_continuous_bullet import \
    ParameterCartPoleContinuousBulletDdpg
from g_utils.types import ModelType


class ParameterComparisonCartPoleBulletDqn(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleBulletEnv-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn(),
            ParameterCartPoleDqn()
        ]


class ParameterComparisonCartPoleBulletDqnTypes(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleBulletEnv-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleBulletDqn(),
            ParameterCartPoleBulletDoubleDqn(),
            ParameterCartPoleBulletDuelingDqn(),
            ParameterCartPoleBulletDoubleDuelingDqn()
        ]

        self.AGENT_LABELS = [
            "DQN",
            "Double DQN",
            "Dueling DQN",
            "Double Dueling DQN",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5


# OnPolicy
class ParameterComparisonCartPoleBulletA2c(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleBulletEnv-v1"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleBulletA2c(),
            ParameterCartPoleBulletA2c(),
            ParameterCartPoleBulletA2c(),
        ]

        self.AGENT_PARAMETERS[0].LEARNING_RATE = 0.001
        self.AGENT_PARAMETERS[1].LEARNING_RATE = 0.0001
        self.AGENT_PARAMETERS[2].LEARNING_RATE = 0.00001
        self.AGENT_LABELS = [
            "DQN (LEARNING_RATE = 0.001)",
            "DQN (LEARNING_RATE = 0.0001)",
            "DQN (LEARNING_RATE = 0.00001)",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5


class ParameterComparisonCartPoleContinuousBulletDdpg(ParameterComparisonBase):
    def __init__(self):
        ParameterComparisonBase.__init__(self)

        self.ENV_NAME = "CartPoleContinuousBulletEnv-v0"

        self.AGENT_PARAMETERS = [
            ParameterCartPoleContinuousBulletDdpg(),
            ParameterCartPoleContinuousBulletDdpg(),
            ParameterCartPoleContinuousBulletDdpg(),
        ]

        self.AGENT_PARAMETERS[0].MODEL_TYPE = ModelType.SMALL_RECURRENT
        self.AGENT_PARAMETERS[1].MODEL_TYPE = ModelType.SMALL_LINEAR
        self.AGENT_PARAMETERS[2].MODEL_TYPE = ModelType.SMALL_LINEAR_2
        self.AGENT_LABELS = [
            "DDPG + GRU",
            "DDPG + Linear",
            "DDPG + Linear_2",
        ]
        self.MAX_TRAINING_STEPS = 50_000
        self.N_RUNS = 5
