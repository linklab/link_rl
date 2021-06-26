import enum

from codes.e_utils.names import EnvironmentName, RLAlgorithmName, DeepLearningModelName, OptimizerName


class RIPEnvRewardType(enum.Enum):
    ORIGINAL = 0
    OLD = 1
    NEW = 2
    UNTIL_TERMINAL_ZERO = 3


class PARAMETERS_GENERAL:
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = "COMMON"
    PYTHON_PATH = None
    EMA_WINDOW = 10
    VERBOSE = True
    MODEL_SAVE = False

    # [MQTT]
    MQTT_SERVER = "localhost"
    MQTT_PORT = 1883
    MQTT_TOPIC_EPISODE_DETAIL = "Episode_Detail"
    MQTT_TOPIC_SUCCESS_DONE = "Success_Done"
    MQTT_TOPIC_FAIL_DONE = "Fail_Done"
    MQTT_TOPIC_TRANSFER_ACK = "Transfer_Ack"
    MQTT_TOPIC_UPDATE_ACK = "Update_Ack"
    MQTT_TOPIC_ACK = "Ack"
    MQTT_LOG = False

    # [ENV]
    NUM_ENVIRONMENTS = 1

    # [WORKER]
    NUM_WORKERS = 1

    # [TRANSFER]
    SOFT_TRANSFER = False
    SOFT_TRANSFER_TAU = 0.3

    # [TARGET_UPDATE]
    SOFT_TARGET_UPDATE = False
    SOFT_TARGET_UPDATE_TAU = 0.3

    ########################################
    ########################################
    ENVIRONMENT_ID = EnvironmentName.PONG_NO_FRAME_SKIP_V4
    RL_ALGORITHM = RLAlgorithmName.DQN_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.DUELING_DQN_CNN
    OPTIMIZER = OptimizerName.ADAM

    TRAIN_STOP_EPISODE_REWARD = None
    TRAIN_STOP_EPISODE_REWARD_STD = 100.0
    STOP_PATIENCE_COUNT = 10

    REPLAY_BUFFER_SIZE = 100000
    TARGET_NET_SYNC_STEP_PERIOD = 1000
    MIN_REPLAY_SIZE_FOR_TRAIN = 2000

    MAX_EPISODE_STEP_AT_PLAY = 10000
    MAX_EPISODE_STEP = 1000
    MAX_GLOBAL_STEP = 1000000
    TRAIN_STEP_FREQ = 4
    AVG_EPISODE_SIZE_FOR_STAT = 50

    N_STEP = 1

    BATCH_SIZE = 32

    #########################################
    #########################################

    # [MLP_DEEP_LEARNING_MODEL]
    HIDDEN_1_SIZE = 128
    HIDDEN_2_SIZE = 128
    HIDDEN_3_SIZE = 128
    # HIDDEN_SIZE_LIST = [128, 128, 128, 256]

    # [CNN_DEEP_LEARNING_MODEL]
    CNN_CRITIC_HIDDEN_1_SIZE = 128
    CNN_CRITIC_HIDDEN_2_SIZE = 128

    # [OPTIMIZATION]
    GAMMA = 0.99 # discount factor

    # [Policy Gradient]
    CRITIC_LOSS_WEIGHT = 0.1
    ENTROPY_LOSS_WEIGHT = 0.005
    CLIP_GRAD = 3.0
    ACTOR_LEARNING_RATE = 0.0001

    # [MODE]
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = True         # Distributed
    MODE_PARAMETERS_TRANSFER = True     # Transfer

    # [TRAINING]
    LEARNING_RATE = 0.001

    # [TRAJECTORY_SAMPLING]
    TRAJECTORY_SAMPLING = True
    TRAJECTORY_LIMIT_SIZE = 200

    # [CUDA]
    CUDA_VISIBLE_DEVICES_NUMBER_LIST = '1, 2'

    RNN_STEP_LENGTH = 2

    WANDB = True

    TEST_PERIOD_EPISODES = 25
    TEST_NUM_EPISODES = 3
    AVG_STEP_SIZE_FOR_TRAIN_LOSS = 50

    EPSILON_INIT = None
    EPSILON_MIN = None
    EPSILON_MIN_STEP = None

    TRAIN_ONLY_AFTER_EPISODE = False

    CURIOSITY_DRIVEN = False
    CURIOSITY_DRIVEN_ETA = 1.0
    CURIOSITY_DRIVEN_BETA = 0.2
    CURIOSITY_DRIVEN_LAMBDA = 0.1

    PERIODIC_MODEL_SAVE = False

    UNIT_TIME = 0.5
