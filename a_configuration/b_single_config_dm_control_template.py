##################################
## DM_CONTROL_CART_POLE_BALANCE ##
##################################
from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_balance import ConfigDmControlCartPoleBalanceA2c
config = ConfigDmControlCartPoleBalanceA2c()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_balance import ConfigDmControlCartPoleBalanceA3c
config = ConfigDmControlCartPoleBalanceA3c()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_balance import ConfigDmControlCartPoleBalancePpo
config = ConfigDmControlCartPoleBalancePpo()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_balance import ConfigDmControlCartPoleBalanceDdpg
config = ConfigDmControlCartPoleBalanceDdpg()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_balance import ConfigDmControlCartPoleBalanceTd3
config = ConfigDmControlCartPoleBalanceTd3()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_balance import ConfigDmControlCartPoleBalanceSac
config = ConfigDmControlCartPoleBalanceSac()


######################################
## DM_CONTROL_CART_POLE_THREE_POLES ##
######################################
from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_three_poles import ConfigDmControlCartPoleThreePolesA2c
config = ConfigDmControlCartPoleThreePolesA2c()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_three_poles import ConfigDmControlCartPoleThreePolesA3c
config = ConfigDmControlCartPoleThreePolesA3c()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_three_poles import ConfigDmControlCartPoleThreePolesPpo
config = ConfigDmControlCartPoleThreePolesPpo()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_three_poles import ConfigDmControlCartPoleThreePolesDdpg
config = ConfigDmControlCartPoleThreePolesDdpg()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_three_poles import ConfigDmControlCartPoleThreePolesTd3
config = ConfigDmControlCartPoleThreePolesTd3()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_three_poles import ConfigDmControlCartPoleThreePolesSac
config = ConfigDmControlCartPoleThreePolesSac()

from a_configuration.b_single_config.dm_control.config_dm_control_cartpole_three_poles import ConfigDmControlCartPoleThreePolesTdmpc
config = ConfigDmControlCartPoleThreePolesTdmpc()


##################################
## DM_CONTROL_CART_POLE_CHEETAH ##
##################################
from a_configuration.b_single_config.dm_control.config_dm_control_cheetah_run import ConfigDmControlCheetahRunA2c
config = ConfigDmControlCheetahRunA2c()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah_run import ConfigDmControlCheetahRunA3c
config = ConfigDmControlCheetahRunA3c()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah_run import ConfigDmControlCheetahRunPpo
config = ConfigDmControlCheetahRunPpo()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/dm_control_cheetah_run_PPO_yhhan/reports/dm_control_cheetah_run_PPO--VmlldzoyMTIwNDM3?accessToken=iznwh3lje5aum3woxmmfk7ce13fhke3k64ap2iqwxjlbuoe5gb0l94eu3hiznt78"

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah_run import ConfigDmControlCheetahRunDdpg
config = ConfigDmControlCheetahRunDdpg()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah_run import ConfigDmControlCheetahRunTd3
config = ConfigDmControlCheetahRunTd3()

from a_configuration.b_single_config.dm_control.config_dm_control_cheetah_run import ConfigDmControlCheetahRunSac
config = ConfigDmControlCheetahRunSac()


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
