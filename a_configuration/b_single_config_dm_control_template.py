##################################
## DM_CONTROL_CART_POLE_BALANCE ##
##################################
from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleBalanceA2c
config = ConfigDmControlCartPoleBalanceA2c()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleBalanceA3c
config = ConfigDmControlCartPoleBalanceA3c()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleBalancePpo
config = ConfigDmControlCartPoleBalancePpo()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleBalanceDdpg
config = ConfigDmControlCartPoleBalanceDdpg()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleBalanceTd3
config = ConfigDmControlCartPoleBalanceTd3()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleBalanceSac
config = ConfigDmControlCartPoleBalanceSac()


######################################
## DM_CONTROL_CART_POLE_THREE_POLES ##
######################################
from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleThreePolesA2c
config = ConfigDmControlCartPoleThreePolesA2c()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleThreePolesA3c
config = ConfigDmControlCartPoleThreePolesA3c()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleThreePolesPpo
config = ConfigDmControlCartPoleThreePolesPpo()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleThreePolesDdpg
config = ConfigDmControlCartPoleThreePolesDdpg()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleThreePolesTd3
config = ConfigDmControlCartPoleThreePolesTd3()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole import ConfigDmControlCartPoleThreePolesSac
config = ConfigDmControlCartPoleThreePolesSac()


##################################
## DM_CONTROL_CART_POLE_CHEETAH ##
##################################
from a_configuration.b_single_config.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahA2c
config = ConfigDmControlCheetahA2c()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahA3c
config = ConfigDmControlCheetahA3c()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahPpo
config = ConfigDmControlCheetahPpo()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahDdpg
config = ConfigDmControlCheetahDdpg()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahTd3
config = ConfigDmControlCheetahTd3()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah import ConfigDmControlCheetahSac
config = ConfigDmControlCheetahSac()


############################################
## DM_CONTROL_CART_POLE_BALL_IN_CUP_CATCH ##
############################################
from a_configuration.b_single_config.dm_control.config_dm_control_ball_in_cup_catch import ConfigDmControlBallInCupCatchA2c
config = ConfigDmControlBallInCupCatchA2c()

from a_configuration.b_single_config.dm_control.config_dm_control_ball_in_cup_catch import ConfigDmControlBallInCupCatchA3c
config = ConfigDmControlBallInCupCatchA3c()

from a_configuration.b_single_config.dm_control.config_dm_control_ball_in_cup_catch import ConfigDmControlBallInCupCatchPpo
config = ConfigDmControlBallInCupCatchPpo()

from a_configuration.b_single_config.dm_control.config_dm_control_ball_in_cup_catch import ConfigDmControlBallInCupCatchDdpg
config = ConfigDmControlBallInCupCatchDdpg()

from a_configuration.b_single_config.dm_control.config_dm_control_ball_in_cup_catch import ConfigDmControlBallInCupCatchTd3
config = ConfigDmControlBallInCupCatchTd3()

from a_configuration.b_single_config.dm_control.config_dm_control_ball_in_cup_catch import ConfigDmControlBallInCupCatchSac
config = ConfigDmControlBallInCupCatchSac()


if __name__ == "__main__":
    from g_utils.commons import print_basic_info, get_env_info, set_config

    set_config(config)
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)