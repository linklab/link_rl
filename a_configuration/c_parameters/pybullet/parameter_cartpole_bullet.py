from a_configuration.b_base.a_environments.pybullet.gym_pybullet import ParameterCartPoleBullet
from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn, \
    ParameterDoubleDqn, ParameterDuelingDqn, ParameterDoubleDuelingDqn
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterPpo
from a_configuration.b_base.parameter_base import ParameterBase
from g_utils.types import ModelType


class ParameterCartPoleBulletDqn(
    ParameterBase, ParameterCartPoleBullet, ParameterDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleBullet.__init__(self)
        ParameterDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleBulletDoubleDqn(
    ParameterBase, ParameterCartPoleBullet, ParameterDoubleDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleBullet.__init__(self)
        ParameterDoubleDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleBulletDuelingDqn(
    ParameterBase, ParameterCartPoleBullet, ParameterDuelingDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleBullet.__init__(self)
        ParameterDuelingDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleBulletDoubleDuelingDqn(
    ParameterBase, ParameterCartPoleBullet, ParameterDoubleDuelingDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleBullet.__init__(self)
        ParameterDoubleDuelingDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleBulletA2c(
    ParameterBase, ParameterCartPoleBullet, ParameterA2c
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleBullet.__init__(self)
        ParameterA2c.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleBulletPpo(
    ParameterBase, ParameterCartPoleBullet, ParameterPpo
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPoleBullet.__init__(self)
        ParameterPpo.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.BATCH_SIZE = 256
        self.PPO_TRAJECTORY_SIZE = self.BATCH_SIZE * 10
        self.BUFFER_CAPACITY = self.PPO_TRAJECTORY_SIZE


