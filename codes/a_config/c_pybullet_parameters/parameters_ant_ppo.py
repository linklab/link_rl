from codes.a_config._rl_parameters.on_policy.parameter_ppo import PARAMETERS_PPO
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_ANT_PPO(PARAMETERS_GENERAL, PARAMETERS_PPO):
    ENVIRONMENT_ID = EnvironmentName.PYBULLET_ANT_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_PPO_V0
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 1500.0
    STOP_PATIENCE_COUNT = 10

    MAX_GLOBAL_STEP     = 10000000
    ACTION_STD_MIN_STEP = 5000000

    GAMMA = 0.99
    AVG_EPISODE_SIZE_FOR_STAT = 50

    ACTION_SCALE = 1.0

    ## PPO
    PPO_GAE_LAMBDA = 0.70
    PPO_TRAJECTORY_SIZE = 2049
    PPO_TRAJECTORY_BATCH_SIZE = 128
    ACTOR_LEARNING_RATE = 0.0002
    LEARNING_RATE = 0.001
    PPO_K_EPOCHS = 10
    PPO_EPSILON_CLIP = 0.2
    N_STEP = 1

    ENTROPY_LOSS_WEIGHT = 0.01
    CLIP_GRAD = 3.0