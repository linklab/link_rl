from codes.a_config._rl_parameters.on_policy.parameter_a2c import PARAMETERS_A2C
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_ANT_A2C(PARAMETERS_GENERAL, PARAMETERS_A2C):
    ENVIRONMENT_ID = EnvironmentName.PYBULLET_ANT_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_STOCHASTIC_ACTOR_CRITIC_MLP
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_A2C_V0
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 2000.0
    STOP_PATIENCE_COUNT = 10

    MAX_GLOBAL_STEP = 30000000
    ACTOR_LEARNING_RATE = 0.0001
    LEARNING_RATE = 0.0001

    GAMMA = 0.99
    BATCH_SIZE = 128

    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1
    TRAIN_STEP_FREQ = 1

    CLIP_GRAD = 3.0
