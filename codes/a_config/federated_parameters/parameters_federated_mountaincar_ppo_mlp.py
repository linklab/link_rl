from codes.a_config._rl_parameters.on_policy.parameter_ppo import PARAMETERS_PPO
from codes.a_config.federated_parameters.parameters_federated_cartpole_ppo_mlp import PARAMETERS_FEDERATED_CARTPOLE_PPO_MLP
from codes.e_utils.names import *


class PARAMETERS_FEDERATED_MOUNTAINCAR_PPO_MLP(PARAMETERS_FEDERATED_CARTPOLE_PPO_MLP, PARAMETERS_PPO):
    # [WORKER]
    NUM_WORKERS = 8

    # [OPTIMIZATION]
    MAX_EPISODES = 5000
    GAMMA = 0.98

    # [MODE]
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = True      # Distributed
    MODE_PARAMETERS_TRANSFER = True    # Transfer

    # [TRAINING]
    EPSILON_INIT = 0.9
    EPSILON_MIN = 0.05
    OPTIMIZER = OptimizerName.ADAM
    PPO_GAE_LAMBDA = 0.99
    LEARNING_RATE = 0.001

    # [TRAJECTORY_SAMPLING]
    TRAJECTORY_SAMPLING = True
    TRAJECTORY_LIMIT_SIZE = 200
    PPO_TRAJECTORY_BATCH_SIZE = 64

    # [PPO]
    PPO_K_EPOCHS = 10
    PPO_EPSILON_CLIP = 0.1

    # [DQN]
    BATCH_SIZE = 128

    # [1. ENVIRONMENTS]
    ENVIRONMENT_ID = EnvironmentName.MOUNTAINCARCONTINUOUS_V0

    # [2. DEEP_LEARNING_MODELS]
    DEEP_LEARNING_MODEL = DeepLearningModelName.ACTOR_CRITIC_MLP

    # [3. ALGORITHMS]
    RL_ALGORITHM = RLAlgorithmName.CONTINUOUS_PPO_V0