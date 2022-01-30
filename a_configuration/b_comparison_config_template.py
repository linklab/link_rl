from g_utils.commons import print_basic_info, get_env_info, print_comparison_basic_info

###############
## CART_POLE ##
###############
from a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import ConfigComparisonCartPoleDqn
config_c = ConfigComparisonCartPoleDqn()

from a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import ConfigComparisonCartPoleDqnTypes
config_c = ConfigComparisonCartPoleDqnTypes()


######################
## CART_POLE_BULLET ##
######################
from a_configuration.c_comparison_config.pybullet.config_comparison_cartpole_bullet import ConfigComparisonCartPoleBulletA2c
config_c = ConfigComparisonCartPoleBulletA2c()

from a_configuration.c_comparison_config.pybullet.config_comparison_cartpole_bullet import ConfigComparisonCartPoleBulletDqnTypes
config_c = ConfigComparisonCartPoleBulletDqnTypes()

from a_configuration.c_comparison_config.pybullet.config_comparison_cartpole_bullet import ConfigComparisonCartPoleContinuousBulletDdpg
config_c = ConfigComparisonCartPoleContinuousBulletDdpg()


######################
## PONG ##
######################
from a_configuration.c_comparison_config.open_ai_gym.config_comparison_pong import ConfigComparisonPongDqn
config_c = ConfigComparisonPongDqn()

from a_configuration.c_comparison_config.open_ai_gym.config_comparison_pong import ConfigComparisonPongDqnTypes
config_c = ConfigComparisonPongDqnTypes()


##################
## LUNAR_LANDER ##
##################
from a_configuration.c_comparison_config.open_ai_gym.config_comparison_lunar_lander import ConfigComparisonLunarLanderDqnRecurrent
config_c = ConfigComparisonLunarLanderDqnRecurrent()


#####################
### HOPPER_MUJOCO ###
#####################
from a_configuration.c_comparison_config.mujoco.config_comparison_hopper import ConfigComparisonHopperMujocoSac
config_c = ConfigComparisonHopperMujocoSac()


#######################
### WALKER2d_MUJOCO ###
#######################
from a_configuration.c_comparison_config.mujoco.config_comparison_walker2d import ConfigComparisonWalker2dMujocoSac
config_c = ConfigComparisonWalker2dMujocoSac()


##########################
### HALFCHEETAH_MUJOCO ###
##########################
from a_configuration.c_comparison_config.mujoco.config_comparison_halfcheetah import ConfigComparisonHalfCheetahMujocoSac
config_c = ConfigComparisonHalfCheetahMujocoSac()


##################
### ANT_MUJOCO ###
##################
from a_configuration.c_comparison_config.mujoco.config_comparison_ant import ConfigComparisonAntMujocoSac
config_c = ConfigComparisonAntMujocoSac()


######################
##    ANT_BULLET    ##
######################
from a_configuration.c_comparison_config.pybullet.config_comparison_ant import ConfigComparisonAntBulletDDpgTd3
config_c = ConfigComparisonAntBulletDDpgTd3()
config_c.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/AntBulletEnv-v0_Comparison_anonymous/reports/AntBullet-DDPG-TD3--VmlldzoxNTAwOTgw?accessToken=1wzn8ux59sr6991ejtvmr1vgq1k1berw6ia2hu4bcm445gn5o1yk43ksbd4qkwhw"

from a_configuration.c_comparison_config.pybullet.config_comparison_ant import ConfigComparisonAntBulletSac
config_c = ConfigComparisonAntBulletSac()


#########################################
##    InvertedDoublePendulum_BULLET    ##
#########################################
from a_configuration.c_comparison_config.pybullet.config_comparison_inverted_double_pendulum_bullet import ConfigComparisonInvertedDoublePendulumBulletSac
config_c = ConfigComparisonInvertedDoublePendulumBulletSac()

from a_configuration.c_comparison_config.pybullet.config_comparison_inverted_double_pendulum_bullet import ConfigComparisonInvertedDoublePendulumBulletA2cPpo
config_c = ConfigComparisonInvertedDoublePendulumBulletA2cPpo()

config_c.USE_WANDB = False

if __name__ == "__main__":
    observation_space, action_space = get_env_info(config_c)
    print_comparison_basic_info(observation_space, action_space, config_c)

