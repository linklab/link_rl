from a_configuration.b_base.a_environments.pybullet.gym_pybullet import ParameterCartPoleBullet, \
    ParameterCartPoleContinuousBullet
from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn, ParameterSac, ParameterDdpg, \
    ParameterDoubleDqn, ParameterDuelingDqn, ParameterDoubleDuelingDqn
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
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
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR)


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
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR)


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
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR)


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
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR)


    # OnPolicy
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
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR)


#############################################################################################

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
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR)


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
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR_2)


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
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR)
        # self.MODEL = ParameterRecurrentLinearModel(ModelType.SMALL_RECURRENT)


