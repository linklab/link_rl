from codes.a_config._rl_parameters.on_policy.parameter_ppo import PARAMETERS_PPO
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_HALF_CHEETAH_PPO(PARAMETERS_GENERAL, PARAMETERS_PPO):
    ENVIRONMENT_ID      = EnvironmentName.PYBULLET_HALF_CHEETAH_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP
    RL_ALGORITHM        = RLAlgorithmName.CONTINUOUS_PPO_V0
    OPTIMIZER           = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 2000.0
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 1000000
    TARGET_NET_SYNC_STEP_PERIOD = 10000
    MAX_GLOBAL_STEP = 10000000
    GAMMA = 0.99
    BATCH_SIZE = 32
    AVG_EPISODE_SIZE_FOR_STAT = 50

    ## PPO
    PPO_GAE_LAMBDA = 0.95
    PPO_TRAJECTORY_SIZE = 2049
    PPO_TRAJECTORY_BATCH_SIZE = 64
    ACTOR_LEARNING_RATE = 0.00002
    LEARNING_RATE = 0.0001
    PPO_K_EPOCHS = 15
    PPO_EPSILON_CLIP = 0.3
    N_STEP = 1