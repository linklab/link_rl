from codes.e_utils.names import EnvironmentName, RLAlgorithmName, DeepLearningModelName, OptimizerName, ModelSaveMode


class PARAMETERS_GENERAL:
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = None
    PYTHON_PATH = None
    EMA_WINDOW = 10
    VERBOSE = True
    MODEL_SAVE = False
    ENV_RENDER = False

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

    # MQTT for RIP
    MQTT_SERVER_FOR_RIP = "192.168.0.10"
    MQTT_PUB_TO_SERVO_POWER = 'motor_power_2'
    MQTT_PUB_RESET = 'reset_2'
    MQTT_SUB_FROM_SERVO = 'servo_info_2'
    MQTT_SUB_MOTOR_LIMIT = 'motor_limit_info_2'
    MQTT_SUB_RESET_COMPLETE = 'reset_complete_2'

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

    STOP_MEAN_EPISODE_REWARD = None
    STOP_PATIENCE_COUNT = 10
    REPLAY_BUFFER_SIZE = 100000
    TARGET_NET_SYNC_STEP_PERIOD = 1000
    MIN_REPLAY_SIZE_FOR_TRAIN = 10000
    EPSILON_MIN_STEP = None
    MAX_GLOBAL_STEP = 1000000
    TRAIN_STEP_FREQ = 4
    AVG_EPISODE_SIZE_FOR_STAT = 50
    N_STEP = 1
    OMEGA = False
    OMEGA_WINDOW_SIZE = 6
    NEXT_STATE_IN_TRAJECTORY = True
    DATA_SAVE_STEP_PERIOD = 1000
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
    ENTROPY_LOSS_WEIGHT = 0.0001
    CLIP_GRAD = 0.5
    ACTOR_LEARNING_RATE = 0.0001

    # [MODE]
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = True         # Distributed
    MODE_PARAMETERS_TRANSFER = True     # Transfer

    # [TRAINING]
    EPSILON_GREEDY_ACT = False
    EPSILON_DECAY = True
    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.05
    OPTIMIZER = OptimizerName.ADAM
    PPO_GAE_LAMBDA = 0.95
    LEARNING_RATE = 0.001

    # [TRAJECTORY_SAMPLING]
    TRAJECTORY_SAMPLING = True
    TRAJECTORY_LIMIT_SIZE = 200
    PPO_TRAJECTORY_BATCH_SIZE = 64

    # [PPO]
    PPO_K_EPOCHS = 10
    PPO_EPSILON_CLIP = 0.2
    PPO_TRAJECTORY_SIZE = 2049

    # [DQN]
    PER_PROPORTIONAL = False
    PER_RANK_BASED = False
    DOUBLE = True

    # [CUDA]
    CUDA_VISIBLE_DEVICES_NUMBER_LIST = '1, 2'

    RNN_STEP_LENGTH = 2

    WANDB = True

    TEST_NUM_EPISODES = 3
    EARLY_STOPPING_TEST_EPISODE_PERIOD = 10
    MODEL_SAVE_MODE = ModelSaveMode.TRAIN
    AVG_STEP_SIZE_FOR_TRAIN_LOSS = 50
