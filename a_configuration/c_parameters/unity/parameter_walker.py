from torch import nn

from a_configuration.b_base.a_environments.unity.unity_box import ParameterWalker
from a_configuration.b_base.b_agents.agents_off_policy import ParameterDdpg
from a_configuration.b_base.parameter_base import ParameterBase
from g_utils.types import ModelType


class ParameterWalkerDdqg(ParameterBase, ParameterWalker, ParameterDdpg):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterWalker.__init__(self)
        ParameterDdpg.__init__(self)

        self.BATCH_SIZE = 64
        self.ACTOR_LEARNING_RATE = 0.0002
        self.LEARNING_RATE = 0.001
        self.N_STEP = 1
        self.BUFFER_CAPACITY = 250_000
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 2_000_000
        self.MODEL_TYPE = ModelType.SMALL_LINEAR