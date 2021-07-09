from codes.a_config._rl_parameters.off_policy.parameter_dqn import PARAMETERS_DQN
from codes.a_config._rl_parameters.off_policy.parameter_td3 import PARAMETERS_TD3, TD3ActionType, TD3ActionSelectorType
from codes.a_config.e_real_device.parameters_quanser_rip_td3 import PARAMETERS_QUANSER_RIP_TD3
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_SYNCRONIZE_TD3(PARAMETERS_GENERAL, PARAMETERS_DQN, PARAMETERS_QUANSER_RIP_TD3):
    ENVIRONMENT_ID = EnvironmentName.ADJUST_ANGLE_V0
    RL_ALGORITHM = RLAlgorithmName.DQN_V0
    DEEP_LEARNING_MODEL = DeepLearningModelName.DUELING_DQN_MLP

    GOAL_ANGLE = 0.0

    UNIT_TIME = 0.006