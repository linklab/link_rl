from a_configuration.b_base.b_agents.agents_off_policy import ParameterDdpg
from a_configuration.b_base.parameter_base import ParameterBase
from a_configuration.b_base.a_environments.unity.unity_box import Parameter3DBall
from g_utils.types import ModelType


class Parameter3DBallDdqg(ParameterBase, Parameter3DBall, ParameterDdpg):
    def __init__(self):
        ParameterBase.__init__(self)
        Parameter3DBall.__init__(self)
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
        self.MODEL = ModelType.SMALL_LINEAR
