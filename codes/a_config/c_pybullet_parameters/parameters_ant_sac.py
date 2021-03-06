from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_ANT_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID      = EnvironmentName.PYBULLET_ANT_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_SAC_MLP
    RL_ALGORITHM        = RLAlgorithmName.CONTINUOUS_SAC_V0
    OPTIMIZER           = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 2000.0
    TRAIN_STOP_EPISODE_REWARD_STD = 50.0
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 1000000

    MAX_GLOBAL_STEP = 10000000
    GAMMA = 0.99
    BATCH_SIZE = 128
    AVG_EPISODE_SIZE_FOR_STAT = 50

    ## SAC
    ACTOR_LEARNING_RATE = 0.0001
    LEARNING_RATE = 0.001
    TRAIN_STEP_FREQ = 2
    N_STEP = 1
