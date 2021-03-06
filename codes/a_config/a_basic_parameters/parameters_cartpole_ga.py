from codes.a_config._rl_parameters.black_box.parameter_ga import GENETIC_NEURO_EVOLUTION_ALGORITHM
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_CARTPOLE_GA(PARAMETERS_GENERAL, GENETIC_NEURO_EVOLUTION_ALGORITHM):
    ENVIRONMENT_ID = EnvironmentName.CARTPOLE_V1
    RL_ALGORITHM = RLAlgorithmName.GENETIC_ALGORITHM
    DEEP_LEARNING_MODEL = DeepLearningModelName.SIMPLE_MLP

    TRAIN_STOP_EPISODE_REWARD = 195.0
    STOP_PATIENCE_COUNT = 10

    LEARNING_RATE = 0.001

    POPULATION_SIZE = 50
    NOISE_STANDARD_DEVIATION = 0.01
