from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC
from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionType, TD3ActionSelectorType, PARAMETERS_TD3
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_HUMANOID_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID = EnvironmentName.PYBULLET_HUMANOID_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_SAC_MLP
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_SAC_V0
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 900.0
    TRAIN_STOP_EPISODE_REWARD_STD = 50.0
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 1000000
    MAX_GLOBAL_STEP = 10000000

    ACTOR_LEARNING_RATE = 0.0002
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    BATCH_SIZE = 128
    TRAIN_STEP_FREQ = 2
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1