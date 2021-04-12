from codes.e_utils.names import DeepLearningModelName, RLAlgorithmName


class PARAMETERS_TD3:
    DEEP_LEARNING_MODEL = DeepLearningModelName.TD3_MLP
    RL_ALGORITHM = RLAlgorithmName.TD3_V0

    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    DOUBLE = True

    ACTION_SCALE = 1.0

    REPLAY_BUFFER_SIZE = None

    ACT_NOISE = 1.0
    NOISE_CLIP = 1.0

    POLICY_UPDATE_FREQUENCY = 2

    TAU = 0.001

    COUNT_BASED_EXPLORATION = False
    N_STEP = 2

    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    EPSILON_MIN_STEP = 1000000

    TRAIN_ONLY_AFTER_EPISODE = True
    NUM_TRAIN_ONLY_AFTER_EPISODE = 100

    TYPE_OF_ACTION = "old"

    TYPE_OF_TD3_ACTION_SELECTOR = "TD3ActionSelector"

    DISTRIBUTIONAL = False
