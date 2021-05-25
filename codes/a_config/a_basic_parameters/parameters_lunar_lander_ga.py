from codes.a_config._rl_parameters.black_box.parameter_ga import GENETIC_NEURO_EVOLUTION_ALGORITHM
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_LUNAR_LANDER_GA(PARAMETERS_GENERAL, GENETIC_NEURO_EVOLUTION_ALGORITHM):
    ENVIRONMENT_ID = EnvironmentName.LUNAR_LANDER_V2
    RL_ALGORITHM = RLAlgorithmName.GENETIC_ALGORITHM
    DEEP_LEARNING_MODEL = DeepLearningModelName.SIMPLE_MLP

    TRAIN_STOP_EPISODE_REWARD = 270.0
    STOP_PATIENCE_COUNT = 10

    LEARNING_RATE = 0.001