from codes.a_config._rl_parameters.parameter_multi_genetic_algorithm import MULTI_GENETIC_NEURO_EVOLUTION_ALGORITHM
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_LUNAR_LANDER_MULTI_GA(PARAMETERS_GENERAL, MULTI_GENETIC_NEURO_EVOLUTION_ALGORITHM):
    ENVIRONMENT_ID = EnvironmentName.LUNAR_LANDER_V2
    DEEP_LEARNING_MODEL = DeepLearningModelName.SIMPLE_MLP
    RL_ALGORITHM = RLAlgorithmName.MULTI_GENETIC_ALGORITHM

    STOP_MEAN_EPISODE_REWARD = 200.0
    STOP_PATIENCE_COUNT = 10

    LEARNING_RATE = 0.001
