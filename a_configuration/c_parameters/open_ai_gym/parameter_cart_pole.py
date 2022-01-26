from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn, ParameterDoubleDqn, ParameterDuelingDqn, \
    ParameterDoubleDuelingDqn
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce, ParameterPpo
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.c_models.recurrent_linear_models import ParameterRecurrentLinearModel
from a_configuration.b_base.parameter_base import ParameterBase
from a_configuration.b_base.a_environments.open_ai_gym.gym_classic_control import ParameterCartPole
from g_utils.types import ModelType


class ParameterCartPoleDqn(
    ParameterBase, ParameterCartPole, ParameterDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleDoubleDqn(
    ParameterBase, ParameterCartPole, ParameterDoubleDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterDoubleDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleDuelingDqn(
    ParameterBase, ParameterCartPole, ParameterDuelingDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterDuelingDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleDoubleDuelingDqn(
    ParameterBase, ParameterCartPole, ParameterDoubleDuelingDqn
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterDoubleDuelingDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 50_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

# OnPolicy

class ParameterCartPoleReinforce(
    ParameterBase, ParameterCartPole, ParameterReinforce
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterReinforce.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.BUFFER_CAPACITY = 1_000
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPoleA2c(
    ParameterBase, ParameterCartPole, ParameterA2c
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterA2c.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BATCH_SIZE = 256
        self.BUFFER_CAPACITY = 10_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterCartPolePpo(
    ParameterBase, ParameterCartPole, ParameterPpo
):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterCartPole.__init__(self)
        ParameterPpo.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.BATCH_SIZE = 256
        self.PPO_TRAJECTORY_SIZE = self.BATCH_SIZE * 10
        self.BUFFER_CAPACITY = self.PPO_TRAJECTORY_SIZE
