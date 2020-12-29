from config.names import OptimizerName, RLAlgorithmName, EnvironmentName, DeepLearningModelName, ReplayBufferName


class PARAMETERS_DOUBLE_RIP:
    # [GENERAL]
    SEED = 1
    MY_PLATFORM = None
    PYTHON_PATH = None
    EMA_WINDOW = 10
    VERBOSE = True
    MODEL_SAVE = False
    ENV_RENDER = False
    CUDA = True

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
    MQTT_SERVER = '192.168.0.10'
    MQTT_PUB_TO_DRIP = 'motor_power'
    MQTT_PUB_RESET = 'reset'
    MQTT_SUB_RESET_COMPLETE = 'reset_complete'
    MQTT_SUB_FROM_DRIP = 'next_state'
    MQTT_ERROR = 'error'

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
    STOP_MEAN_EPISODE_REWARD = 1000
    STOP_CONDITION_CONTINUOUS_EPISODE = 10

    REPLAY_BUFFER_SIZE = 100000
    MIN_REPLAY_SIZE_FOR_TRAIN = 1000
    TARGET_NET_SYNC_STEP_PERIOD = 1000
    MAX_GLOBAL_STEP = 5000000
    TRAIN_STEP_FREQ = 4
    AVG_EPISODE_SIZE_FOR_STAT = 10
    MODEL_SAVE_STEP_PERIOD = 10000
    DRAW_VIZ = True
    DRAW_VIZ_PERIOD_STEPS = 10
    N_STEP = 1
    OMEGA = False
    OMEGA_WINDOW_SIZE = 6
    NEXT_STATE_IN_TRAJECTORY = True
    DATA_SAVE_STEP_PERIOD = 1000

    LOAD_SAVED_ACTOR_MODEL = None    # 정확하고도 완전한 Full File PATH + NAME 지정 (확장자 포함)
    LOAD_SAVED_CRITIC_MODEL = None   # 정확하고도 완전한 Full File PATH + NAME 지정 (확장자 포함)
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
    GAMMA = 0.98 # discount factor

    # [Policy Gradient]
    ENTROPY_BETA = 0.01
    CLIP_GRAD = 0.1
    ACTOR_LEARNING_RATE = 0.0001

    # [MODE]
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = True         # Distributed
    MODE_PARAMETERS_TRANSFER = True     # Transfer

    # [TRAINING]
    EPSILON_GREEDY_ACT = False
    EPSILON_DECAY = True
    EPSILON_DECAY_RATE = 1000 #Large value means low decaying

    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.001
    EPSILON_MIN_STEP = 1000000

    EPSILON_SWING_UP_MIN_STEP = 500000
    EPSILON_BALANCING_MIN_STEP = 200000

    PER = False
    CH = False
    OPTIMIZER = OptimizerName.ADAM
    GAE_LAMBDA = 0.95
    LEARNING_RATE = 0.001
    ACTION_SCALE = SWING_UP_SCALE_FACTOR = 200
    BALANCING_SCALE_FACTOR = 0.01
    ENV_RESET = False

    # [TRAJECTORY_SAMPLING]
    TRAJECTORY_SAMPLING = True
    TRAJECTORY_LIMIT_SIZE = 200
    TRAJECTORY_BATCH_SIZE = 64

    # [PPO]
    PPO_K_EPOCH = 10
    PPO_EPSILON_CLIP = 0.2
    PPO_VALUE_LOSS_WEIGHT = 0.5
    PPO_ENTROPY_WEIGHT = 0.01

    # [DQN]
    BATCH_SIZE = 128

    # [CUDA]
    CUDA_VISIBLE_DEVICES_NUMBER_LIST = '2, 3'



    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.REAL_DEVICE_DOUBLE_RIP

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.ACTOR_CRITIC_MLP

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.DDPG_FAST_V0

    # [4. OPTIMIZER]
    OPTIMIZER = OptimizerName.ADAM