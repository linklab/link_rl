import enum
import os
idx = os.getcwd().index("{0}link_rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "link_rl{0}".format(os.sep)


class OSName(enum.Enum):
    MAC = "MAC"
    WINDOWS = "WINDOWS"
    LINUX = "LINUX"


class EnvironmentName(enum.Enum):
    CARTPOLE_V0 = "CartPole-v0"
    CARTPOLE_V1 = "CartPole-v1"
    MOUNTAINCARCONTINUOUS_V0 = "MountainCarContinuous-v0"
    ACROBOT_V1 = "Acrobot-v1"
    BLACKJACK_V0 = "Blackjack-v0"
    QUANSER_SERVO_2 = "Quanser_Servo_2"
    CHASER_V1_MAC = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Chaser_v1")
    CHASER_V1_WINDOWS = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Chaser_v1.exe")
    BREAKOUT_DETERMINISTIC_V4 = "BreakoutDeterministic-v4"
    PENDULUM_V0 = 'Pendulum-v0'
    DRONE_RACING_MAC = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "DroneEnv_forMac")
    DRONE_RACING_WINDOWS = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Dron_Racing.exe")
    GRIDWORLD_V0 = 'Gridworld-v0'
    FROZENLAKE_V0 = 'FrozenLake-v0'
    INVERTED_DOUBLE_PENDULUM_V2 = 'InvertedDoublePendulum-v2'
    HOPPER_V2 = 'Hopper-v2'
    ANT_V2 = 'Ant-v2'
    HALF_CHEETAH_V2 = 'HalfCheetah-v2'
    SWIMMER_V2 = 'Swimmer-v2'
    REACHER_V2 = 'Reacher-v2'
    HUMANOID_V2 = 'Humanoid-v2'
    HUMANOID_STAND_UP_V2 = 'HumanoidStandup-v2'
    INVERTED_PENDULUM_V2 = 'InvertedPendulum-v2'
    WALKER_2D_V2 = 'Walker2d-v2'
    PONG_NO_FRAME_SKIP_V4 = 'PongNoFrameskip-v4'
    BREAKOUT_NO_FRAME_SKIP_V4 = 'BreakoutNoFrameskip-v4'
    SPACE_INVADERS_NO_FRAME_SKIP_V4 = "SpaceInvadersNoFrameskip-v4"
    PENDULUM_MATLAB_V0 = "Pendulum_Matlab_v0"


class DeepLearningModelName(enum.Enum):
    Actor_Critic_MLP = "Actor_Critic_MLP"
    Actor_Critic_CNN = "Actor_Critic_CNN"
    No_Model = "No_Model"
    Dueling_DQN_CNN = "Dueling_DQN_CNN"
    Dueling_DQN_MLP = "Dueling_DQN_MLP"


class RLAlgorithmName(enum.Enum):
    DQN_V0 = "DQN_v0"
    DQN_FAST_V0 = "DQN_FAST_V0"   # FAST_RL
    DDPG_FAST_V0 = "DDPG_FAST_V0"  # FAST_RL
    PPO_V0 = "PPO_v0"
    Policy_Iteration = "DP_Policy_Iteration"
    Value_Iteration = "DP_Value_Iteration"
    Monte_Carlo_Control_V0 = "Monte_Carlo_Control_v0"


class OptimizerName(enum.Enum):
    NESTEROV = "nesterov"
    ADAM = "Adam"


class ReplayBufferName(enum.Enum):
    REPLAY_BUFFER = "Replay_Buffer"
    PRIORITIZED_REPLAY_BUFFER = "Prioritized_Replay_Buffer"
