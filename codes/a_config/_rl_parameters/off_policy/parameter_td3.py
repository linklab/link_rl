from codes.e_utils.names import DeepLearningModelName, RLAlgorithmName


class PARAMETERS_TD3:
    DEEP_LEARNING_MODEL = DeepLearningModelName.TD3_MLP
    RL_ALGORITHM = RLAlgorithmName.TD3_V0

    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    DOUBLE = True

    EPSILON_INIT = None
    EPSILON_MIN = None
    EPSILON_MIN_STEP = None

    REPLAY_BUFFER_SIZE = None

    ACT_NOISE = 0.1
    NOISE_CLIP = 0.5
    TARGET_NOISE = 0.2
