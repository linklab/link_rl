from codes.a_config._rl_parameters.on_policy.parameter_sac import PARAMETERS_SAC
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_ANT_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID      = EnvironmentName.PYBULLET_ANT_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.SOFT_ACTOR_CRITIC_MLP
    RL_ALGORITHM        = RLAlgorithmName.SAC_V0
    OPTIMIZER           = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 1500.0
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 1000000
    TARGET_NET_SYNC_STEP_PERIOD = 10000
    MAX_GLOBAL_STEP = 70000000
    GAMMA = 0.99
    BATCH_SIZE = 32
    AVG_EPISODE_SIZE_FOR_STAT = 50

    ACTION_SCALE = 1.0

    ## SAC
    ACTOR_LEARNING_RATE = 0.00001
    LEARNING_RATE = 0.0001
    TRAIN_STEP_FREQ = 1
    N_STEP = 1

    PER = False