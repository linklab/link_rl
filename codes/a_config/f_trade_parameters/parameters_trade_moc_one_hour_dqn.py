from codes.a_config.f_trade_parameters.parameters_trade_dqn import PARAMETERS_GENERAL_TRADE_DQN
from codes.b_environments.trade.trade_constant import TimeUnit
from codes.e_utils.names import *
from codes.a_config.parameters_general import PARAMETERS_GENERAL


class PARAMETERS_TRADE_MOC_ONE_HOUR_DQN(PARAMETERS_GENERAL_TRADE_DQN):
    COIN_NAME = "MOC"
    TIME_UNIT = TimeUnit.ONE_HOUR