from a_configuration.c_parameters.Unity.parameter_3dball import Parameter3DBallDdqg
from a_configuration.c_parameters.mujoco.parameter_halfcheetah_mujoco import ParameterHalfCheetahMujocoSac
from a_configuration.c_parameters.mujoco.parameter_hopper_mujoco import ParameterHopperMujocoSac
from a_configuration.c_parameters.mujoco.parameter_walker2d_mujoco import ParameterWalker2dMujocoSac
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn, \
    ParameterCartPoleDoubleDqn, \
    ParameterCartPoleDuelingDqn, ParameterCartPoleDoubleDuelingDqn
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleReinforce
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleA2c
from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderContinuousA2c, \
    ParameterLunarLanderA2c, ParameterLunarLanderContinuousDdpg, ParameterLunarLanderContinuousSac
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDqn, ParameterPongDoubleDqn, \
    ParameterPongDuelingDqn, ParameterPongDoubleDuelingDqn
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongA2c
from a_configuration.c_parameters.mujoco.parameter_ant_mujoco import ParameterAntMujocoSac

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletDqn, \
    ParameterCartPoleContinuousBulletA2c, ParameterCartPoleContinuousBulletSac, ParameterCartPoleContinuousBulletDdpg, \
    ParameterCartPoleBulletDoubleDqn, ParameterCartPoleBulletDuelingDqn, ParameterCartPoleBulletDoubleDuelingDqn
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

parameter_cart_pole_ddqn = ParameterCartPoleDoubleDqn()
parameter_list.append(parameter_cart_pole_ddqn)

parameter_cart_pole_duelingdqn = ParameterCartPoleDuelingDqn()
parameter_list.append(parameter_cart_pole_duelingdqn)

parameter_cart_pole_doubleduelingdqn = ParameterCartPoleDoubleDuelingDqn()
parameter_list.append(parameter_cart_pole_doubleduelingdqn)

parameter_cart_pole_reinforce = ParameterCartPoleReinforce()
parameter_list.append(parameter_cart_pole_reinforce)

parameter_cart_pole_a2c = ParameterCartPoleA2c()
parameter_list.append(parameter_cart_pole_a2c)

#############################
## LUNAR_LANDER ##
#############################
parameter_lunar_lander_a2c = ParameterLunarLanderA2c()
parameter_list.append(parameter_lunar_lander_a2c)

parameter_lunar_lander_continuous_a2c = ParameterLunarLanderContinuousA2c()
parameter_list.append(parameter_lunar_lander_continuous_a2c)

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

parameter_cart_pole_continuous_bullet_a2c = ParameterCartPoleContinuousBulletA2c()
parameter_list.append(parameter_cart_pole_continuous_bullet_a2c)

parameter_cart_pole_continuous_bullet_sac = ParameterCartPoleContinuousBulletSac()
parameter_list.append(parameter_cart_pole_continuous_bullet_sac)

parameter_cart_pole_continuous_bullet_ddpg = ParameterCartPoleContinuousBulletDdpg()
parameter_list.append(parameter_cart_pole_continuous_bullet_ddpg)

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
parameter_hopper_mujoco_sac = ParameterHopperMujocoSac()
parameter_list.append(parameter_hopper_mujoco_sac)


#######################
### WALKER2d_MUJOCO ###
#######################
parameter_walker2d_mujoco_sac = ParameterWalker2dMujocoSac()
parameter_list.append(parameter_walker2d_mujoco_sac)


##########################
### HALFCHEETAH_MUJOCO ###
##########################
parameter_halfcheetah_mujoco_sac = ParameterHalfCheetahMujocoSac()
parameter_list.append(parameter_halfcheetah_mujoco_sac)


##########################
### Unity3DBall ###
##########################
parameter_3d_ball_ddpg = Parameter3DBallDdqg()
parameter_list.append(parameter_3d_ball_ddpg)

###########################################################
for parameter in parameter_list:
    parameter.USE_WANDB = False
    parameter.WANDB_ENTITY = "link-koreatech"


if __name__ == "__main__":
    parameter = parameter_cart_pole_duelingdqn
    print_basic_info(parameter=parameter)
