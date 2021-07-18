from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_HALF_CHEETAH_DDPG(PARAMETERS_GENERAL, PARAMETERS_DDPG):
    ENVIRONMENT_ID      = EnvironmentName.PYBULLET_HALF_CHEETAH_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_DETERMINISTIC_ACTOR_CRITIC_MLP
    RL_ALGORITHM        = RLAlgorithmName.DDPG_V0
    OPTIMIZER           = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 500.0
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 1000000
    TARGET_NET_SYNC_STEP_PERIOD = 10000
    MAX_GLOBAL_STEP = 10000000
    LEARNING_RATE = 0.00025
    GAMMA = 0.99
    BATCH_SIZE = 128
    TRAIN_STEP_FREQ = 4
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1
