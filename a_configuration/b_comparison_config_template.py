from g_utils.commons import get_env_info, print_comparison_basic_info

###############
## CART_POLE ##
###############
from a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import ConfigComparisonCartPoleDqn
config_c = ConfigComparisonCartPoleDqn()

from a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import ConfigComparisonCartPoleDqnTypes
config_c = ConfigComparisonCartPoleDqnTypes()

from a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import ConfigComparisonCartPolePpo
config_c = ConfigComparisonCartPolePpo()

######################
## CART_POLE_BULLET ##
######################

from a_configuration.c_comparison_config.pybullet.config_comparison_cart_pole_bullet import ConfigComparisonCartPoleBulletDqn
config_c = ConfigComparisonCartPoleBulletDqn()

from a_configuration.c_comparison_config.pybullet.config_comparison_cart_pole_bullet import ConfigComparisonCartPoleBulletA2c
config_c = ConfigComparisonCartPoleBulletA2c()

from a_configuration.c_comparison_config.pybullet.config_comparison_cart_pole_bullet import ConfigComparisonCartPoleBulletDqnTypes
config_c = ConfigComparisonCartPoleBulletDqnTypes()

#################################
## CART_POLE_CONTINUOUS_BULLET ##
#################################

from a_configuration.c_comparison_config.pybullet.config_comparison_cart_pole_continuous_bullet import ConfigComparisonCartPoleContinuousBulletDdpg
config_c = ConfigComparisonCartPoleContinuousBulletDdpg()

from a_configuration.c_comparison_config.pybullet.config_comparison_cart_pole_continuous_bullet import ConfigComparisonCartPoleContinuousBulletAll
config_c = ConfigComparisonCartPoleContinuousBulletAll()

from a_configuration.c_comparison_config.pybullet.config_comparison_cart_pole_continuous_bullet import ConfigComparisonCartPoleContinuousBulletPpo
config_c = ConfigComparisonCartPoleContinuousBulletPpo()

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

from a_configuration.c_comparison_config.mujoco.config_comparison_hopper import ConfigComparisonHopperMujocoSacPpo
config_c = ConfigComparisonHopperMujocoSacPpo()

#######################
### WALKER2d_MUJOCO ###
#######################
from a_configuration.c_comparison_config.mujoco.config_comparison_walker2d import ConfigComparisonWalker2dMujocoSac
config_c = ConfigComparisonWalker2dMujocoSac()

from a_configuration.c_comparison_config.mujoco.config_comparison_walker2d import ConfigComparisonWalker2dMujocoSacPpo
config_c = ConfigComparisonWalker2dMujocoSacPpo()

##########################
### HALFCHEETAH_MUJOCO ###
##########################
from a_configuration.c_comparison_config.mujoco.config_comparison_halfcheetah import ConfigComparisonHalfCheetahMujocoSac
config_c = ConfigComparisonHalfCheetahMujocoSac()

from a_configuration.c_comparison_config.mujoco.config_comparison_halfcheetah import ConfigComparisonHalfCheetahMujocoSacPpo
config_c = ConfigComparisonHalfCheetahMujocoSacPpo()

##################
### ANT_MUJOCO ###
##################
from a_configuration.c_comparison_config.mujoco.config_comparison_ant import ConfigComparisonAntMujocoSac
config_c = ConfigComparisonAntMujocoSac()

from a_configuration.c_comparison_config.mujoco.config_comparison_ant import ConfigComparisonAntMujocoSacPpo
config_c = ConfigComparisonAntMujocoSacPpo()

######################
##    ANT_BULLET    ##
######################
from a_configuration.c_comparison_config.pybullet.config_comparison_ant import ConfigComparisonAntBulletDDpgTd3
config_c = ConfigComparisonAntBulletDDpgTd3()
config_c.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/AntBulletEnv-v0_Comparison_anonymous/reports/AntBullet-DDPG-TD3--VmlldzoxNTAwOTgw?accessToken=1wzn8ux59sr6991ejtvmr1vgq1k1berw6ia2hu4bcm445gn5o1yk43ksbd4qkwhw"

from a_configuration.c_comparison_config.pybullet.config_comparison_ant import ConfigComparisonAntBulletSacAlpha
config_c = ConfigComparisonAntBulletSacAlpha()

from a_configuration.c_comparison_config.pybullet.config_comparison_ant import ConfigComparisonAntBulletSacPer
config_c = ConfigComparisonAntBulletSacPer()

from a_configuration.c_comparison_config.pybullet.config_comparison_ant import ConfigComparisonAntBulletPpoSac
config_c = ConfigComparisonAntBulletPpoSac()

#########################################
##    InvertedDoublePendulum_BULLET    ##
#########################################
from a_configuration.c_comparison_config.pybullet.config_comparison_inverted_double_pendulum_bullet import ConfigComparisonInvertedDoublePendulumBulletSacAlpha
config_c = ConfigComparisonInvertedDoublePendulumBulletSacAlpha()

from a_configuration.c_comparison_config.pybullet.config_comparison_inverted_double_pendulum_bullet import ConfigComparisonInvertedDoublePendulumBulletA2cPpo
config_c = ConfigComparisonInvertedDoublePendulumBulletA2cPpo()

config_c.USE_WANDB = False

if __name__ == "__main__":
    observation_space, action_space = get_env_info(config_c)
    print_comparison_basic_info(observation_space, action_space, config_c)

