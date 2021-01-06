from config.fast_rl_parameters.parameters_fast_rl_breakout_dqn          import PARAMETERS_FAST_RL_BREAKOUT_DQN
from config.fast_rl_parameters.parameters_fast_rl_cartpole_a2c          import PARAMETERS_FAST_RL_CARTPOLE_A2C
from config.fast_rl_parameters.parameters_fast_rl_pong_dqn              import PARAMETERS_FAST_RL_PONG_DQN
from config.fast_rl_parameters.parameters_fast_rl_cartpole_dqn          import PARAMETERS_FAST_RL_CARTPOLE_DQN
from config.fast_rl_parameters.parameters_fast_rl_cartpole_pg           import PARAMETERS_FAST_RL_CARTPOLE_PG
from config.fast_rl_parameters.parameters_fast_rl_cartpole_reinforce    import PARAMETERS_FAST_RL_CARTPOLE_REINFORCE
from config.fast_rl_parameters.parameters_fast_rl_minitaur_bullet_a2c   import PARAMETERS_FAST_RL_MINITAUR_BULLET_A2C
from config.fast_rl_parameters.parameters_fast_rl_pendulum_matlab_dqn   import PARAMETERS_FAST_RL_PENDULUM_MATLAB_DQN

from config.fast_rl_parameters.parameters_fast_rl_pendulum_a2c_continuous_action import PARAMETERS_FAST_RL_PENDULUM_A2C_CONTINUOUS_ACTION
from config.fast_rl_parameters.parameters_fast_rl_pendulum_ddpg                  import PARAMETERS_FAST_RL_PENDULUM_DDPG
from config.fast_rl_parameters.parameters_fast_rl_pendulum_d4pg                  import PARAMETERS_FAST_RL_PENDULUM_D4PG
from config.federated_parameters.parameters_federated_matlab_rip_ddpg import PARAMETERS_FEDERATED_MATLAB_RIP_DDPG
from config.federated_parameters.parameters_federated_pendulum_ddpg import \
    PARAMETERS_FEDERATED_PENDULUM_DDPG
from config.names import DeepLearningModelName, RLAlgorithmName, EnvironmentName

from config.or_parameters.parameters_fast_rl_knapsack_dqn import PARAMETERS_FAST_RL_KNAPSACK_DQN
from config.or_parameters.parameters_fast_rl_tsp_dqn import PARAMETERS_FAST_RL_TSP_DQN

from config.fast_rl_parameters.parameters_fast_rl_pendulum_matlab_ddpg import PARAMETERS_FAST_RL_PENDULUM_MATLAB_DDPG

from config.federated_parameters.parameters_federated_matlab_double_rip_ddpg import PARAMETERS_FEDERATED_MATLAB_DOUBLE_RIP_DDPG
from config.federated_parameters.parameters_federated_cartpole_ppo_mlp import PARAMETERS_FEDERATED_CARTPOLE_PPO_MLP
from config.real_device.parameters_double_rip import PARAMETERS_DOUBLE_RIP
from config.names import *


#
# class PARAMETERS(PARAMETERS_FEDERATED_CARTPOLE_PPO_MLP):
#     MY_PLATFORM = OSName.REAL_RIP
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     DRAW_VIZ = False
#     ENV_RENDER = False
#     MODE_SYNCHRONIZATION = True
#     MODE_GRADIENTS_UPDATE = False  # Distributed
#     MODE_PARAMETERS_TRANSFER = False  # Transfer

class PARAMETERS(PARAMETERS_DOUBLE_RIP):
    MY_PLATFORM = OSName.REAL_RIP
    PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
    ENV_RENDER = False
    NUM_WORKERS = 1
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = True  # Distributed
    MODE_PARAMETERS_TRANSFER = False  # Transfer

# class PARAMETERS(PARAMETERS_FEDERATED_PENDULUM_DDPG):
#     MY_PLATFORM = OSName.REAL_RIP
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     ENV_RENDER = False
#     NUM_WORKERS = 4
#     MODE_SYNCHRONIZATION = True
#     MODE_GRADIENTS_UPDATE = True      # Distributed
#     MODE_PARAMETERS_TRANSFER = False    # Transfer

# class PARAMETERS(PARAMETERS_FEDERATED_MATLAB_RIP_DDPG):
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     ENV_RENDER = False
#     NUM_WORKERS = 1
#     MODE_SYNCHRONIZATION = True
#     MODE_GRADIENTS_UPDATE = True      # Distributed
#     MODE_PARAMETERS_TRANSFER = False    # Transfer
#     ENV_RESET = False
#     ENVIRONMENT_ID = EnvironmentName.PENDULUM_MATLAB_DOUBLE_AGENTS_V0
#     RL_ALGORITHM = RLAlgorithmName.DDPG_FAST_DOUBLE_AGENTS_V0
    # ENVIRONMENT_ID = EnvironmentName.PENDULUM_MATLAB_V0
    # RL_ALGORITHM = RLAlgorithmName.DDPG_FAST_V0


# class PARAMETERS(PARAMETERS_FEDERATED_MATLAB_DOUBLE_RIP_DDPG):
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     DRAW_VIZ = False
#     PER = False

# class PARAMETERS(PARAMETERS_FAST_RL_PENDULUM_DDPG):
#     DRAW_VIZ = False
#     PER = False

# class PARAMETERS(PARAMETERS_FAST_RL_PENDULUM_MATLAB_DDPG):
#     DRAW_VIZ = False
#     ENV_RESET = False

# class PARAMETERS(PARAMETERS_FAST_RL_CARTPOLE_DQN):
#     DRAW_VIZ = False