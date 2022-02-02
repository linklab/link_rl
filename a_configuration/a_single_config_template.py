###############
## CART_POLE ##
###############
from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()

from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDoubleDqn
config = ConfigCartPoleDoubleDqn()

from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDuelingDqn
config = ConfigCartPoleDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleDoubleDuelingDqn
config = ConfigCartPoleDoubleDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleReinforce
config = ConfigCartPoleReinforce()

from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPoleA2c
config = ConfigCartPoleA2c()

from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPolePpo
config = ConfigCartPolePpo()

from a_configuration.b_single_config.open_ai_gym.config_cart_pole import ConfigCartPolePpoTrajectory
config = ConfigCartPolePpoTrajectory()

######################
## CART_POLE_BULLET ##
######################
from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletDqn
config = ConfigCartPoleBulletDqn()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletDoubleDqn
config = ConfigCartPoleBulletDoubleDqn()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletDuelingDqn
config = ConfigCartPoleBulletDuelingDqn()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletDoubleDuelingDqn
config = ConfigCartPoleBulletDoubleDuelingDqn()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletA2c
config = ConfigCartPoleBulletA2c()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletPpo
config = ConfigCartPoleBulletPpo()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletPpoTrajectory
config = ConfigCartPoleBulletPpoTrajectory()

#################################
## CART_POLE_CONTINUOUS_BULLET ##
#################################
from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletA2c
config = ConfigCartPoleContinuousBulletA2c()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletPpo
config = ConfigCartPoleContinuousBulletPpo()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletPpoTrajectory
config = ConfigCartPoleContinuousBulletPpoTrajectory()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletSac
config = ConfigCartPoleContinuousBulletSac()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletDdpg
config = ConfigCartPoleContinuousBulletDdpg()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletTd3
config = ConfigCartPoleContinuousBulletTd3()

##################
## LUNAR_LANDER ##
##################
from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderDqn
config = ConfigLunarLanderDqn()

from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderA2c
config = ConfigLunarLanderA2c()

from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderPpo
config = ConfigLunarLanderPpo()

from a_configuration.b_single_config.open_ai_gym.config_lunar_lander import ConfigLunarLanderPpoTrajectory
config = ConfigLunarLanderPpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLander-v2_PPO_TRAJECTORY_anonymous/reports/LunarLander-v2-PPO_TRAJECTORY--VmlldzoxNTAwOTg0?accessToken=y2tgu6i38zzehklrq6rikdqhvza2fsexmfg22ntl2drl0q9bsvd11t1b0e09r0ry"

#############################
## LUNAR_LANDER_CONTINUOUS ##
#############################
from a_configuration.b_single_config.open_ai_gym.config_lunar_lander_continuous import ConfigLunarLanderContinuousA2c
config = ConfigLunarLanderContinuousA2c()

from a_configuration.b_single_config.open_ai_gym.config_lunar_lander_continuous import ConfigLunarLanderContinuousPpo
config = ConfigLunarLanderContinuousPpo()

from a_configuration.b_single_config.open_ai_gym.config_lunar_lander_continuous import ConfigLunarLanderContinuousPpoTrajectory
config = ConfigLunarLanderContinuousPpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_PPO_TRAJECTORY_yhhan/reports/LunarLanderContinuous-v2_PPO_TRAJECTORY--VmlldzoxNTA3MjAx"

from a_configuration.b_single_config.open_ai_gym.config_lunar_lander_continuous import ConfigLunarLanderContinuousDdpg
config = ConfigLunarLanderContinuousDdpg()

from a_configuration.b_single_config.open_ai_gym.config_lunar_lander_continuous import ConfigLunarLanderContinuousTd3
config = ConfigLunarLanderContinuousTd3()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_TD3_yhhan/reports/LunarLanderContinuous-v2_TD3--VmlldzoxNTA3MjA4"

from a_configuration.b_single_config.open_ai_gym.config_lunar_lander_continuous import ConfigLunarLanderContinuousSac
config = ConfigLunarLanderContinuousSac()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_SAC_yhhan/reports/LunarLanderContinuous-v2_SAC--VmlldzoxNTA3MjA2"

######################
##    ANT_BULLET    ##
######################
from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletA2c
config = ConfigAntBulletA2c()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletPpo
config = ConfigAntBulletPpo()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletPpoTrajectory
config = ConfigAntBulletPpoTrajectory()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletDdpg
config = ConfigAntBulletDdpg()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletTd3
config = ConfigAntBulletTd3()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletSac
config = ConfigAntBulletSac()

#########################
##    HOPPER_BULLET    ##
#########################
from a_configuration.b_single_config.pybullet.config_hopper_bullet import ConfigHopperBulletSac
config = ConfigHopperBulletSac()

from a_configuration.b_single_config.pybullet.config_hopper_bullet import ConfigHopperBulletPpoTrajectory
config = ConfigHopperBulletPpoTrajectory()

#########################################
##    InvertedDoublePendulum_BULLET    ##
#########################################
from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletA2c
config = ConfigInvertedDoublePendulumBulletA2c()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletSac
config = ConfigInvertedDoublePendulumBulletSac()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletDdpg
config = ConfigInvertedDoublePendulumBulletDdpg()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletTd3
config = ConfigInvertedDoublePendulumBulletTd3()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletPpo
config = ConfigInvertedDoublePendulumBulletPpo()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletPpoTrajectory
config = ConfigInvertedDoublePendulumBulletPpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/InvertedDoublePendulumBulletEnv-v0_PPO_TRAJECTORY_anonymous/reports/Inverted-Double-Pendulum-PPO_Trajectory--VmlldzoxNTAwNzg4?accessToken=j639pzfajkaiybgddbxkp552v0qjcv6zu0ogytbr6ec85qxg7j2gefdh56gvvdx7"

##########
## PONG ##
##########
from a_configuration.b_single_config.open_ai_gym.config_pong import ConfigPongDqn
config = ConfigPongDqn()

from a_configuration.b_single_config.open_ai_gym.config_pong import ConfigPongDoubleDqn
config = ConfigPongDoubleDqn()

from a_configuration.b_single_config.open_ai_gym.config_pong import ConfigPongDuelingDqn
config = ConfigPongDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.config_pong import ConfigPongDoubleDuelingDqn
config = ConfigPongDoubleDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.config_pong import ConfigPongA2c
config = ConfigPongA2c()


##################
### ANT_MUJOCO ###
##################
from a_configuration.b_single_config.mujoco.config_ant_mujoco import ConfigAntMujocoSac
config = ConfigAntMujocoSac()

from a_configuration.b_single_config.mujoco.config_ant_mujoco import ConfigAntMujocoPpoTrajectory
config = ConfigAntMujocoPpoTrajectory()

#####################
### HOPPER_MUJOCO ###
#####################
from a_configuration.b_single_config.mujoco.config_hopper_mujoco import ConfigHopperMujocoSac
config = ConfigHopperMujocoSac()

from a_configuration.b_single_config.mujoco.config_hopper_mujoco import ConfigHopperMujocoPpoTrajectory
config = ConfigHopperMujocoPpoTrajectory()

#######################
### WALKER2d_MUJOCO ###
#######################
from a_configuration.b_single_config.mujoco.config_walker2d_mujoco import ConfigWalker2dMujocoSac
config = ConfigWalker2dMujocoSac()

from a_configuration.b_single_config.mujoco.config_walker2d_mujoco import ConfigWalker2dMujocoPpoTrajectory
config = ConfigWalker2dMujocoPpoTrajectory()

##########################
### HALFCHEETAH_MUJOCO ###
##########################
from a_configuration.b_single_config.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoSac
config = ConfigHalfCheetahMujocoSac()

from a_configuration.b_single_config.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoPpoTrajectory
config = ConfigHalfCheetahMujocoPpoTrajectory()

##########################
### Unity3DBall ###
##########################
from a_configuration.b_single_config.unity.config_3d_ball import Config3DBallDdqg
config = Config3DBallDdqg()


##########################
### UnityWalker ###
##########################
from a_configuration.b_single_config.unity.config_walker import ConfigWalkerDdqg
config = ConfigWalkerDdqg()

config.USE_WANDB = False

if __name__ == "__main__":
    from g_utils.commons import print_basic_info, get_env_info
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)