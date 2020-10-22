from config.names import OptimizerName, RLAlgorithmName, EnvironmentName, DeepLearningModelName, ReplayBufferName


class PARAMETERS_GENERAL:
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
    MQTT_SERVER_FOR_RIP = "192.168.0.10"
    MQTT_PUB_TO_SERVO_POWER = 'motor_power_2'
    MQTT_PUB_RESET = 'reset_2'
    MQTT_SUB_FROM_SERVO = 'servo_info_2'
    MQTT_SUB_MOTOR_LIMIT = 'motor_limit_info_2'
    MQTT_SUB_RESET_COMPLETE = 'reset_complete_2'

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
    RL_ALGORITHM = RLAlgorithmName.DQN_FAST_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.DUELING_DQN_CNN
    REPLAY_BUFFER = ReplayBufferName.REPLAY_BUFFER

    STOP_MEAN_EPISODE_REWARD = None
    REPLAY_BUFFER_SIZE = 100000
    MIN_REPLAY_SIZE_FOR_TRAIN = 1000
    TARGET_NET_SYNC_STEP_PERIOD = 1000
    EPSILON_MIN_STEP = 1000
    MAX_GLOBAL_STEPS = 1000000
    TRAIN_STEP_FREQ = 4
    AVG_EPISODE_SIZE_FOR_STAT = 100
    MODEL_SAVE_STEP_PERIOD = 10000
    DRAW_VIZ = True
    DRAW_VIZ_PERIOD_STEPS = 10
    N_STEP = 1
    OMEGA = False
    OMEGA_WINDOW_SIZE = 6
    NEXT_STATE_IN_TRAJECTORY = True
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
    MAX_EPISODES = 2000
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
    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.05
    EPSILON_DECAY_RATE = 1000 #Large value means low decaying
    OPTIMIZER = OptimizerName.ADAM
    GAE_LAMBDA = 0.95
    LEARNING_RATE = 0.001

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