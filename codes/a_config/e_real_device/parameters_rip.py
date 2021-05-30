from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG
from codes.e_utils.names import OptimizerName, RLAlgorithmName, EnvironmentName, DeepLearningModelName
from codes.a_config.parameters_general import PARAMETERS_GENERAL

class PARAMETERS_RIP(PARAMETERS_GENERAL, PARAMETERS_DDPG):
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = None
    PYTHON_PATH = None
    EMA_WINDOW = 10
    VERBOSE = True
    MODEL_SAVE = False
    ########################################
    ########################################
    TRAIN_STOP_EPISODE_REWARD = 1000
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 100000
    TARGET_NET_SYNC_STEP_PERIOD = 1000
    MAX_GLOBAL_STEP = 5000000
    TRAIN_STEP_FREQ = 4
    AVG_EPISODE_SIZE_FOR_STAT = 10
    N_STEP = 4
    OMEGA = False
    OMEGA_WINDOW_SIZE = 6

    #########################################
    #########################################

    # [MLP_DEEP_LEARNING_MODEL]
    HIDDEN_1_SIZE = 128
    HIDDEN_2_SIZE = 128
    HIDDEN_3_SIZE = 128
    # HIDDEN_SIZE_LIST = [128, 128, 128, 256]

    # [OPTIMIZATION]
    GAMMA = 0.98 # discount factor

    # [Policy Gradient]
    ENTROPY_LOSS_WEIGHT = 0.01
    CLIP_GRAD = 0.1
    ACTOR_LEARNING_RATE = 0.0001

    # [TRAINING]
    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.001
    EPSILON_MIN_STEP = 1000000

    PER = False
    LEARNING_RATE = 0.001
    ACTION_SCALE = 200
    BALANCING_SCALE_FACTOR = 0.01
    ENV_RESET = False

    # [DQN]
    BATCH_SIZE = 32

    # [CUDA]
    CUDA_VISIBLE_DEVICES_NUMBER_LIST = '1, 2'

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.REAL_DEVICE_RIP

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.DDPG_V0

    # [4. OPTIMIZER]
    OPTIMIZER = OptimizerName.ADAM

    UNIT_TIME = 0.006

    PENDULUM_STATE_INFO = 0  # 1: ALL ARM INFO IGNORED, 2: ARM ANGLE INFO IGNORED (VELOCITY IS INCLUDED)
