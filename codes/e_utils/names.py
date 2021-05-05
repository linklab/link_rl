import enum
import os, sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)


class AgentMode(enum.Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    PLAY = "PLAY"


class OSName(enum.Enum):
    MAC = "MAC"
    WINDOWS = "WINDOWS"
    LINUX = "LINUX"
    REAL_RIP = "REAL_RIP_PLATFORM"


class EnvironmentName(enum.Enum):
    ACROBOT_V1 = "Acrobot-v1"
    CARTPOLE_V0 = "CartPole-v0"
    CARTPOLE_V1 = "CartPole-v1"
    MOUNTAINCAR_V0 = "MountainCar-v0"
    MOUNTAINCARCONTINUOUS_V0 = "MountainCarContinuous-v0"
    BLACKJACK_V0 = "Blackjack-v0"
    QUANSER_SERVO_2 = "Quanser_Servo_2"
    CHASER_V1_MAC = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Chaser_v1")
    CHASER_V1_WINDOWS = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Chaser_v1.exe")
    BREAKOUT_DETERMINISTIC_V4 = "BreakoutDeterministic-v4"
    PENDULUM_V0 = 'Pendulum-v0'
    LUNAR_LANDER_V2 = 'LunarLander-v2'
    LUNAR_LANDER_CONTINUOUS_V2 = "LunarLanderContinuous-v2"
    DRONE_RACING_MAC = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "DroneEnv_forMac")
    DRONE_RACING_WINDOWS = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Dron_Racing.exe")
    GRIDWORLD_V0 = 'Gridworld-v0'
    FROZENLAKE_V0 = 'FrozenLake-v0'
    INVERTED_DOUBLE_PENDULUM_V2 = 'InvertedDoublePendulum-v2'
    HOPPER_V2 = 'Hopper-v2'
    PYBULLET_ANT_V0 = 'AntPyBulletEnv-v0'
    PYBULLET_INVERTED_DOUBLE_PENDULUM_V0 = 'InvertedDoublePendulumPyBulletEnv-v0'
    PYBULLET_HALF_CHEETAH_V0 = 'HalfCheetahPyBulletEnv-v0'
    SWIMMER_V2 = 'Swimmer-v2'
    REACHER_V2 = 'Reacher-v2'
    HUMANOID_V2 = 'Humanoid-v2'
    HUMANOID_STAND_UP_V2 = 'HumanoidStandup-v2'
    INVERTED_PENDULUM_V2 = 'InvertedPendulum-v2'
    WALKER_2D_V2 = 'Walker2d-v2'
    PONG_NO_FRAME_SKIP_V4 = 'PongNoFrameskip-v4'
    KUNGFU_MASTER_FRAME_SKIP_V4 = 'KungFuMasterNoFrameskip-v4'
    BREAKOUT_NO_FRAME_SKIP_V4 = 'BreakoutNoFrameskip-v4'
    SPACE_INVADERS_NO_FRAME_SKIP_V4 = "SpaceInvadersNoFrameskip-v4"
    ENDURO_NO_FRAME_SKIP_V4 = "EnduroNoFrameskip-v4"
    SEAQUEST_NO_FRAME_SKIP_V4 = "SeaquestNoFrameskip-v4"
    FREEWAY_NO_FRAME_SKIP_V4 = "FreewayNoFrameskip-v4"
    TSP_V0 = "TSP-v0"  # bi-directional connections and uniform cost.
    TSP_V1 = "TSP-v1"  # bi-directional connections
    KNAPSACK_V0 = "Knapsack-v0"  # unbounded knapsack problem with 200 items.
    KNAPSACK_V1 = "Knapsack-v1"  # binary (0-1) knapsack problem with 200 items.
    KNAPSACK_V2 = "Knapsack-v2"  # bounded knapsack problem with 200 items.
    KNAPSACK_V3 = "Knapsack-v3"  # stochastic, online knapsack with 200 items.
    PENDULUM_MATLAB_V0 = "Pendulum_Matlab_v0"
    PENDULUM_MATLAB_DOUBLE_RIP_V0 = "Pendulum_Matlab_Double_RIP_v0"
    REAL_DEVICE_RIP = "Real_Device_Rip"
    REAL_DEVICE_DOUBLE_RIP = "Real_Device_Double_Rip"
    TRADE_V0 = "Trade_v0"


class DeepLearningModelName(enum.Enum):
    STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP = "STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP"
    STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP = "STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP"

    STOCHASTIC_DISCRETE_ACTOR_CRITIC_CNN = "STOCHASTIC_DISCRETE_ACTOR_CRITIC_CNN"
    STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_CNN = "STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_CNN"

    SOFT_ACTOR_CRITIC_MLP = "SOFT_ACTOR_CRITIC_MLP"

    DUELING_DQN_MLP = "DUELING_DQN_MLP"
    DUELING_DQN_CNN = "DUELING_DQN_CNN"
    DUELING_DQN_SMALL_CNN = "DUELING_DQN_SMALL_CNN"

    RAINBOW_DQN_MLP = "RAINBOW_DQN_MLP"

    DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP = "DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP"
    DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU = "DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU"
    DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU_ATTENTION = "DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU_ATTENTION"

    TD3_MLP = "TD3_MLP"

    SIMPLE_MLP = "SIMPLE_MLP"
    SIMPLE_CNN = "SIMPLE_CNN"
    SIMPLE_SMALL_CNN = "SIMPLE_SMALL_CNN"


class RLAlgorithmName(enum.Enum):
    DQN_V0 = "DQN_V0"
    RAINBOW_V0 = "RAINBOW_V0"

    DDPG_V0 = "DDPG_V0"
    D4PG_V0 = "D4PG_V0"
    TD3_V0 = "TD3_V0"

    SAC_V0 = "SAC_V0"

    DISCRETE_A2C_V0 = "DISCRETE_A2C_V0"
    CONTINUOUS_A2C_V0 = "CONTINUOUS_A2C_V0"

    POLICY_GRADIENT_V0 = "POLICY_GRADIENT_V0"
    REINFORCE_V0 = "REINFORCE_V0"

    DISCRETE_PPO_V0 = "DISCRETE_PPO_V0"
    CONTINUOUS_PPO_V0 = "CONTINUOUS_PPO_V0"

    EVOLUTION_STRATEGY = "EVOLUTION_STRATEGY"
    GENETIC_ALGORITHM = "GENETIC_ALGORITHM"
    MULTI_GENETIC_ALGORITHM = "MULTI_GENETIC_ALGORITHM"


class OptimizerName(enum.Enum):
    NESTEROV = "nesterov"
    ADAM = "Adam"
    RMSProp = "RMSProp"


OFF_POLICY_RL_ALGORITHMS = [
    RLAlgorithmName.DQN_V0,
    RLAlgorithmName.DDPG_V0,
    RLAlgorithmName.D4PG_V0,
    RLAlgorithmName.RAINBOW_V0,
    RLAlgorithmName.TD3_V0
]

ON_POLICY_RL_ALGORITHMS = [
    RLAlgorithmName.DISCRETE_A2C_V0,
    RLAlgorithmName.CONTINUOUS_A2C_V0,
    RLAlgorithmName.DISCRETE_PPO_V0,
    RLAlgorithmName.CONTINUOUS_PPO_V0,
    RLAlgorithmName.SAC_V0,
]