from codes.a_config.federated_parameters.parameters_federated_cartpole_dqn import PARAMETERS_FEDERATED_CARTPOLE_DQN

#
# class PARAMETERS(PARAMETERS_FEDERATED_CARTPOLE_PPO_MLP):
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     ENV_RENDER = False
#     MODE_SYNCHRONIZATION = True
#     MODE_GRADIENTS_UPDATE = False  # Distributed
#     MODE_PARAMETERS_TRANSFER = False  # Transfer

# class PARAMETERS(PARAMETERS_FEDERATED_PENDULUM_DDPG):
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     ENV_RENDER = False
#     NUM_WORKERS = 1
#     MODE_SYNCHRONIZATION = True
#     MODE_GRADIENTS_UPDATE = True      # Distributed
#     MODE_PARAMETERS_TRANSFER = False    # Transfer

# class PARAMETERS(PARAMETERS_FEDERATED_PENDULUM_PPO):
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     ENV_RENDER = False
#     NUM_WORKERS = 1
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

# class PARAMETERS(PARAMETERS_FEDERATED_MATLAB_RIP_DQN):
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     ENV_RENDER = False
#     NUM_WORKERS = 1
#     MODE_SYNCHRONIZATION = True
#     MODE_GRADIENTS_UPDATE = True      # Distributed
#     MODE_PARAMETERS_TRANSFER = False    # Transfer
#     ENV_RESET = False

class PARAMETERS(PARAMETERS_FEDERATED_CARTPOLE_DQN):
    PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
    ENV_RENDER = True
    NUM_WORKERS = 1
    MODE_SYNCHRONIZATION = True
    MODE_GRADIENTS_UPDATE = False  # Distributed
    MODE_PARAMETERS_TRANSFER = False  # Transfer


# class PARAMETERS(PARAMETERS_FEDERATED_MATLAB_RIP_D4PG):
#     PYTHON_PATH = "~/anaconda3/envs/rl/bin/python"
#     ENV_RENDER = False
#     NUM_WORKERS = 1
#     MODE_SYNCHRONIZATION = True
#     MODE_GRADIENTS_UPDATE = True      # Distributed
#     MODE_PARAMETERS_TRANSFER = False    # Transfer
#     ENV_RESET = False


# class PARAMETERS(PARAMETERS_PENDULUM_DDPG):
#     PER = False

# class PARAMETERS(PARAMETERS_PENDULUM_D4PG):
#     PER = False

# class PARAMETERS(PARAMETERS_PENDULUM_PPO):
#     PER = False

# class PARAMETERS(PARAMETERS_PENDULUM_MATLAB_DDPG):
#     ENV_RESET = False

# class PARAMETERS(PARAMETERS_CARTPOLE_DQN):
#     pass