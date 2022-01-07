from a_configuration.b_base.a_environments.pybullet.gym_mujoco import ParameterAntMujoco, ParameterHopperMujoco
from a_configuration.b_base.a_environments.pybullet.gym_pybullet import ParameterAntBullet
from a_configuration.b_base.b_agents.agents_off_policy import ParameterSac
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.parameter_base import ParameterBase
from g_utils.types import ModelType
from torch import nn

class ParameterHopperMujocoSac(ParameterBase, ParameterHopperMujoco, ParameterSac):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterHopperMujoco.__init__(self)
        ParameterSac.__init__(self)

        self.BATCH_SIZE = 256

        self.ALPHA_LEARNING_RATE = 0.000005
        self.ACTOR_LEARNING_RATE = 0.0002
        self.LEARNING_RATE = 0.001
        self.N_STEP = 2
        self.BUFFER_CAPACITY = 1_000_000
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 2_000_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000

        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR_2)

        self.LAYER_ACTIVATION = nn.ReLU()
        self.LAYER_NORM = False
