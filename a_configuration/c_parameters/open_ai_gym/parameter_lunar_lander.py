from a_configuration.b_base.a_environments.open_ai_gym.gym_box2d import ParameterLunarLander
from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterPpo
from a_configuration.b_base.parameter_base import ParameterBase
from g_utils.types import ModelType


class ParameterLunarLanderDqn(ParameterBase, ParameterLunarLander, ParameterDqn):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterLunarLander.__init__(self)
        ParameterDqn.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 100_000
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterLunarLanderA2c(ParameterBase, ParameterLunarLander, ParameterA2c):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterLunarLander.__init__(self)
        ParameterA2c.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 100_000
        self.BATCH_SIZE = 256
        self.MODEL_TYPE = ModelType.SMALL_LINEAR


class ParameterLunarLanderPpo(ParameterBase, ParameterLunarLander, ParameterPpo):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterLunarLander.__init__(self)
        ParameterPpo.__init__(self)

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 100_000
        self.BUFFER_CAPACITY = 100_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR

        self.BATCH_SIZE = 256
        self.PPO_TRAJECTORY_SIZE = self.BATCH_SIZE * 10
        self.PPO_K_EPOCH = 3
        self.BUFFER_CAPACITY = self.PPO_TRAJECTORY_SIZE
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 10 * 3
