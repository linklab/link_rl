from codes.a_config._rl_parameters.off_policy.parameter_sac import PARAMETERS_SAC
from codes.a_config._rl_parameters.on_policy.parameter_ppo import PARAMETERS_PPO
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL

# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_gym_env.py
class PARAMETERS_HALF_CHEETAH_SAC(PARAMETERS_GENERAL, PARAMETERS_SAC):
    ENVIRONMENT_ID      = EnvironmentName.PYBULLET_HALF_CHEETAH_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.CONTINUOUS_SAC_MLP
    RL_ALGORITHM        = RLAlgorithmName.CONTINUOUS_SAC_V0
    OPTIMIZER           = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = 2000.0
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 1000000

    MAX_GLOBAL_STEP = 10000000
    GAMMA = 0.99

    BATCH_SIZE = 128
    AVG_EPISODE_SIZE_FOR_STAT = 50