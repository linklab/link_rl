from a_configuration.b_base.a_environments.pybullet.gym_pybullet import ParameterAntBullet, ParameterHopperBullet, \
    ParameterHumanoidBullet
from a_configuration.b_base.b_agents.agents_off_policy import ParameterDqn, ParameterDdpg, ParameterSac
from a_configuration.b_base.b_agents.agents_on_policy import ParameterA2c, ParameterReinforce
from a_configuration.b_base.c_models.linear_models import ParameterLinearModel
from a_configuration.b_base.parameter_base import ParameterBase
from g_utils.commons import print_basic_info, get_env_info
from g_utils.types import ModelType


class ParameterHumanoidBulletSac(ParameterBase, ParameterHumanoidBullet, ParameterSac):
    def __init__(self):
        ParameterBase.__init__(self)
        ParameterHopperBullet.__init__(self)
        ParameterSac.__init__(self)

        self.ALPHA_LEARNING_RATE = 0.0001
        self.ACTOR_LEARNING_RATE = 0.0002
        self.LEARNING_RATE = 0.001

        self.BATCH_SIZE = 64

        self.N_STEP = 1
        self.BUFFER_CAPACITY = 250_000
        self.MIN_BUFFER_SIZE_FOR_TRAIN = self.BATCH_SIZE * 10

        self.N_VECTORIZED_ENVS = 1
        self.N_ACTORS = 1
        self.MAX_TRAINING_STEPS = 2_000_000
        self.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 100
        self.TEST_INTERVAL_TRAINING_STEPS = 5_000
        self.MODEL = ParameterLinearModel(ModelType.SMALL_LINEAR_2)


if __name__ == "__main__":
    parameter = ParameterHumanoidBulletSac()
    observation_space, action_space = get_env_info(parameter)
    print_basic_info(observation_space=observation_space, action_space=action_space, parameter=parameter)
