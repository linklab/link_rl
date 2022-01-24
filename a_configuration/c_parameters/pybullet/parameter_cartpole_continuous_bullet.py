from a_configuration.b_base.a_environments.pybullet.gym_pybullet import ParameterCartPoleContinuousBullet
from a_configuration.b_base.b_agents.agents_off_policy import ParameterSac, ParameterDdpg, ParameterTd3
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterPpo
from a_configuration.b_base.parameter_base import ParameterBase
from g_utils.types import ModelType


class ParameterCartPoleContinuousBulletA2c(
    ParameterBase, ParameterCartPoleContinuousBullet, ParameterA2c
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleContinuousBullet.__init__(self)
        ParameterA2c.__init__(self)

        self.LEARNING_RATE = 0.001
        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.BATCH_SIZE = 512
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleContinuousBulletPpo(
    ParameterBase, ParameterCartPoleContinuousBullet, ParameterPpo
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleContinuousBullet.__init__(self)
        ParameterPpo.__init__(self)

        self.LEARNING_RATE = 0.001
        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.BATCH_SIZE = 256
        self.PPO_TRAJECTORY_SIZE = self.BATCH_SIZE * 10
        self.BUFFER_CAPACITY = self.PPO_TRAJECTORY_SIZE


class ParameterCartPoleContinuousBulletSac(
    ParameterBase, ParameterCartPoleContinuousBullet, ParameterSac
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleContinuousBullet.__init__(self)
        ParameterSac.__init__(self)

        self.ALPHA_LEARNING_RATE = 0.0001
        self.LEARNING_RATE = 0.001
        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleContinuousBulletDdpg(
    ParameterBase, ParameterCartPoleContinuousBullet, ParameterDdpg
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleContinuousBullet.__init__(self)
        ParameterDdpg.__init__(self)

        self.LEARNING_RATE = 0.001
        self.N_VECTORIZED_ENVS = 1
        self.BUFFER_CAPACITY = 200_000
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        # self.MODEL_TYPE = ModelType.SMALL_RECURRENT


class ParameterCartPoleContinuousBulletTd3(
    ParameterBase, ParameterCartPoleContinuousBullet, ParameterTd3
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleContinuousBullet.__init__(self)
        ParameterDdpg.__init__(self)

        self.LEARNING_RATE = 0.001
        self.N_VECTORIZED_ENVS = 1
        self.BUFFER_CAPACITY = 200_000
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 200_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR
        # self.MODEL_TYPE = ModelType.SMALL_RECURRENT
