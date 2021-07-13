from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionType, TD3ActionSelectorType, PARAMETERS_TD3
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_HUMANOID_TD3(PARAMETERS_GENERAL, PARAMETERS_TD3):
    ENVIRONMENT_ID = EnvironmentName.PYBULLET_HUMANOID_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.TD3_MLP
    RL_ALGORITHM = RLAlgorithmName.TD3_V0
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 900.0
    TRAIN_STOP_EPISODE_REWARD_STD = 50.0
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 300000
    MAX_GLOBAL_STEP = 10000000

    ACTOR_LEARNING_RATE = 0.0004
    LEARNING_RATE = 0.0004
    GAMMA = 0.99
    BATCH_SIZE = 256
    TRAIN_STEP_FREQ = 1
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 2
    TAU = 0.003

    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    EPSILON_MIN_STEP = 3000000

    TYPE_OF_TD3_ACTION = TD3ActionType.GAUSSIAN_NOISE
    TYPE_OF_TD3_ACTION_SELECTOR = TD3ActionSelectorType.BASIC_ACTION_SELECTOR
