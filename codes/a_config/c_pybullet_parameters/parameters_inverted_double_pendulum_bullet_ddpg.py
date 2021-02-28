from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_INVERTED_DOUBLE_PENDULUM_BULLET_DDPG(PARAMETERS_GENERAL):
    ENVIRONMENT_ID      = EnvironmentName.PYBULLET_INVERTED_DOUBLE_PENDULUM_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP
    RL_ALGORITHM        = RLAlgorithmName.DDPG_V0
    OPTIMIZER           = OptimizerName.ADAM

    STOP_MEAN_EPISODE_REWARD = 900
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 1000000
    TARGET_NET_SYNC_STEP_PERIOD = 10000
    MAX_GLOBAL_STEP = 10000000
    EPSILON_MIN_STEP = 1000000
    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.001
    LEARNING_RATE = 0.00025
    GAMMA = 0.99
    BATCH_SIZE = 32
    TRAIN_STEP_FREQ = 1
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1

    RNN_STEP_LENGTH = 2

    ACTION_SCALE = 1.0

    PER_PROPORTIONAL = False
    PER_RANK_BASED = False