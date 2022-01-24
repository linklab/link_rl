from a_configuration.c_parameters.pybullet.parameter_cartpole_continuous_bullet import \
    ParameterCartPoleContinuousBulletA2c, ParameterCartPoleContinuousBulletPpo, ParameterCartPoleContinuousBulletSac, \
    ParameterCartPoleContinuousBulletDdpg, ParameterCartPoleContinuousBulletTd3
from a_configuration.c_parameters.unity.parameter_3d_ball import Parameter3DBallDdqg
from a_configuration.c_parameters.mujoco.parameter_halfcheetah_mujoco import ParameterHalfCheetahMujocoSac
from a_configuration.c_parameters.mujoco.parameter_hopper_mujoco import ParameterHopperMujocoSac
from a_configuration.c_parameters.mujoco.parameter_walker2d_mujoco import ParameterWalker2dMujocoSac
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn, \
    ParameterCartPoleDoubleDqn, ParameterCartPoleDuelingDqn, ParameterCartPoleDoubleDuelingDqn, ParameterCartPolePpo
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleReinforce
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleA2c
from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderContinuousA2c, \
    ParameterLunarLanderA2c, ParameterLunarLanderContinuousDdpg, ParameterLunarLanderContinuousSac, \
    ParameterLunarLanderDqn, ParameterLunarLanderPpo, ParameterLunarLanderContinuousPpo
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDqn, ParameterPongDoubleDqn, \
    ParameterPongDuelingDqn, ParameterPongDoubleDuelingDqn
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongA2c
from a_configuration.c_parameters.mujoco.parameter_ant_mujoco import ParameterAntMujocoSac

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletDqn, \
    ParameterCartPoleBulletDoubleDqn, ParameterCartPoleBulletDuelingDqn, ParameterCartPoleBulletDoubleDuelingDqn, \
    ParameterCartPoleBulletA2c, ParameterCartPoleBulletPpo
from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletA2c, ParameterAntBulletSac
from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletDdpg
from a_configuration.c_parameters.pybullet.parameter_double_inverted_pendulum_bullet import \
    ParameterDoubleInvertedPendulumBulletSac
from a_configuration.c_parameters.pybullet.parameter_hopper_bullet import ParameterHopperBulletSac
from g_utils.commons import print_basic_info

parameter_list = []

###############
## CART_POLE ##
###############
parameter_cart_pole_dqn = ParameterCartPoleDqn()
parameter_list.append(parameter_cart_pole_dqn)

parameter_cart_pole_double_dqn = ParameterCartPoleDoubleDqn()
parameter_list.append(parameter_cart_pole_double_dqn)

parameter_cart_pole_dueling_dqn = ParameterCartPoleDuelingDqn()
parameter_list.append(parameter_cart_pole_dueling_dqn)

parameter_cart_pole_double_dueling_dqn = ParameterCartPoleDoubleDuelingDqn()
parameter_list.append(parameter_cart_pole_double_dueling_dqn)

parameter_cart_pole_reinforce = ParameterCartPoleReinforce()
parameter_list.append(parameter_cart_pole_reinforce)

parameter_cart_pole_a2c = ParameterCartPoleA2c()
parameter_list.append(parameter_cart_pole_a2c)

parameter_cart_pole_ppo = ParameterCartPolePpo()
parameter_list.append(parameter_cart_pole_ppo)

##################
## LUNAR_LANDER ##
##################

parameter_lunar_lander_dqn = ParameterLunarLanderDqn()
parameter_list.append(parameter_lunar_lander_dqn)

parameter_lunar_lander_a2c = ParameterLunarLanderA2c()
parameter_list.append(parameter_lunar_lander_a2c)

parameter_lunar_lander_ppo = ParameterLunarLanderPpo()
parameter_list.append(parameter_lunar_lander_ppo)

parameter_lunar_lander_continuous_a2c = ParameterLunarLanderContinuousA2c()
parameter_list.append(parameter_lunar_lander_continuous_a2c)

parameter_lunar_lander_continuous_ppo = ParameterLunarLanderContinuousPpo()
parameter_list.append(parameter_lunar_lander_continuous_ppo)

parameter_lunar_lander_continuous_ddpg = ParameterLunarLanderContinuousDdpg()
parameter_list.append(parameter_lunar_lander_continuous_ddpg)

parameter_lunar_lander_continuous_sac = ParameterLunarLanderContinuousSac()
parameter_list.append(parameter_lunar_lander_continuous_sac)

######################
## CART_POLE_BULLET ##
######################
parameter_cart_pole_bullet_dqn = ParameterCartPoleBulletDqn()
parameter_list.append(parameter_cart_pole_bullet_dqn)

parameter_cart_pole_bullet_double_dqn = ParameterCartPoleBulletDoubleDqn()
parameter_list.append(parameter_cart_pole_bullet_double_dqn)

parameter_cart_pole_bullet_dueling_dqn = ParameterCartPoleBulletDuelingDqn()
parameter_list.append(parameter_cart_pole_bullet_dueling_dqn)

parameter_cart_pole_bullet_double_dueling_dqn = ParameterCartPoleBulletDoubleDuelingDqn()
parameter_list.append(parameter_cart_pole_bullet_double_dueling_dqn)

parameter_cart_pole_bullet_a2c = ParameterCartPoleBulletA2c()
parameter_list.append(parameter_cart_pole_bullet_a2c)

parameter_cart_pole_bullet_ppo = ParameterCartPoleBulletPpo()
parameter_list.append(parameter_cart_pole_bullet_ppo)


#################################
## CART_POLE_CONTINUOUS_BULLET ##
#################################

parameter_cart_pole_continuous_bullet_a2c = ParameterCartPoleContinuousBulletA2c()
parameter_list.append(parameter_cart_pole_continuous_bullet_a2c)

parameter_cart_pole_continuous_bullet_ppo = ParameterCartPoleContinuousBulletPpo()
parameter_list.append(parameter_cart_pole_continuous_bullet_ppo)

parameter_cart_pole_continuous_bullet_sac = ParameterCartPoleContinuousBulletSac()
parameter_list.append(parameter_cart_pole_continuous_bullet_sac)

parameter_cart_pole_continuous_bullet_ddpg = ParameterCartPoleContinuousBulletDdpg()
parameter_list.append(parameter_cart_pole_continuous_bullet_ddpg)

parameter_cart_pole_continuous_bullet_td3 = ParameterCartPoleContinuousBulletTd3()
parameter_list.append(parameter_cart_pole_continuous_bullet_td3)

######################
##    ANT_BULLET    ##
######################
parameter_ant_bullet_a2c = ParameterAntBulletA2c()
parameter_list.append(parameter_ant_bullet_a2c)

parameter_ant_bullet_ddpg = ParameterAntBulletDdpg()
parameter_list.append(parameter_ant_bullet_ddpg)

parameter_ant_bullet_sac = ParameterAntBulletSac()
parameter_list.append(parameter_ant_bullet_sac)

#########################
##    HOPPER_BULLET    ##
#########################
parameter_hopper_bullet_sac = ParameterHopperBulletSac()
parameter_list.append(parameter_hopper_bullet_sac)


#########################################
##    DoubleInvertedPendulum_BULLET    ##
#########################################
parameter_double_inverted_pendulum_bullet_sac = ParameterDoubleInvertedPendulumBulletSac()
parameter_list.append(parameter_double_inverted_pendulum_bullet_sac)


##########
## PONG ##
##########
parameter_pong_dqn = ParameterPongDqn()
parameter_list.append(parameter_pong_dqn)

parameter_pong_double_dqn = ParameterPongDoubleDqn()
parameter_list.append(parameter_pong_double_dqn)

parameter_pong_dueling_dqn = ParameterPongDuelingDqn()
parameter_list.append(parameter_pong_dueling_dqn)

parameter_pong_double_dueling_dqn = ParameterPongDoubleDuelingDqn()
parameter_list.append(parameter_pong_double_dueling_dqn)

parameter_pong_a2c = ParameterPongA2c()
parameter_list.append(parameter_pong_a2c)


##################
### ANT_MUJOCO ###
##################
parameter_ant_mujoco_sac = ParameterAntMujocoSac()
parameter_list.append(parameter_ant_mujoco_sac)


#####################
### HOPPER_MUJOCO ###
#####################
parameter = ParameterHopperMujocoSac()

#######################
### WALKER2d_MUJOCO ###
#######################
parameter_walker2d_mujoco_sac = ParameterWalker2dMujocoSac()

##########################
### HALFCHEETAH_MUJOCO ###
##########################
parameter_halfcheetah_mujoco_sac = ParameterHalfCheetahMujocoSac()

##########################
### Unity3DBall ###
##########################
parameter_3d_ball_ddpg = Parameter3DBallDdqg()

###########################################################


if __name__ == "__main__":
    parameter = parameter_cart_pole_dueling_dqn
    print_basic_info(parameter=parameter)
