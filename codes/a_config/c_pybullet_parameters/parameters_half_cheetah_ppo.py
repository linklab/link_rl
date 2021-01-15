from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_HALF_CHEETAH_PPO(PARAMETERS_GENERAL):
    ENVIRONMENT_ID      = EnvironmentName.HALF_CHEETAH_V2
    DEEP_LEARNING_MODEL = DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP
    RL_ALGORITHM        = RLAlgorithmName.CONTINUOUS_PPO_V0
    OPTIMIZER           = OptimizerName.ADAM

    STOP_MEAN_EPISODE_REWARD = 2000.0
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 1000000
    TARGET_NET_SYNC_STEP_PERIOD = 10000
    MAX_GLOBAL_STEP = 10000000
    GAMMA = 0.99
    BATCH_SIZE = 32
    AVG_EPISODE_SIZE_FOR_STAT = 100

    RNN_STEP_LENGTH = 2

    ACTION_SCALE = 1.0

    ## PPO
    PPO_GAE_LAMBDA = 0.95
    PPO_TRAJECTORY_SIZE = 2049
    PPO_TRAJECTORY_BATCH_SIZE = 64
    ACTOR_LEARNING_RATE = 0.00002
    LEARNING_RATE = 0.0001
    PPO_K_EPOCHS = 15
    PPO_EPSILON_CLIP = 0.3
    PPO_ENTROPY_WEIGHT = 0.01
    TRAIN_STEP_FREQ = 1
    N_STEP = 1