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
from a_configuration.c_parameters.unity.parameter_walker import ParameterWalkerDdqg
from g_utils.commons import print_basic_info

###############
## CART_POLE ##
###############
parameter = ParameterCartPoleDqn()

parameter = ParameterCartPoleDoubleDqn()

parameter = ParameterCartPoleDuelingDqn()

parameter = ParameterCartPoleDoubleDuelingDqn()

parameter = ParameterCartPoleReinforce()

parameter = ParameterCartPoleA2c()

parameter = ParameterCartPolePpo()

##################
## LUNAR_LANDER ##
##################

parameter = ParameterLunarLanderDqn()

parameter = ParameterLunarLanderA2c()

parameter = ParameterLunarLanderPpo()

parameter = ParameterLunarLanderContinuousA2c()

parameter = ParameterLunarLanderContinuousPpo()

parameter = ParameterLunarLanderContinuousDdpg()

parameter = ParameterLunarLanderContinuousSac()

######################
## CART_POLE_BULLET ##
######################
parameter = ParameterCartPoleBulletDqn()

parameter = ParameterCartPoleBulletDoubleDqn()

parameter = ParameterCartPoleBulletDuelingDqn()

parameter = ParameterCartPoleBulletDoubleDuelingDqn()

parameter = ParameterCartPoleBulletA2c()

parameter = ParameterCartPoleBulletPpo()


#################################
## CART_POLE_CONTINUOUS_BULLET ##
#################################

parameter = ParameterCartPoleContinuousBulletA2c()

parameter = ParameterCartPoleContinuousBulletPpo()

parameter = ParameterCartPoleContinuousBulletSac()

parameter = ParameterCartPoleContinuousBulletDdpg()

parameter = ParameterCartPoleContinuousBulletTd3()

######################
##    ANT_BULLET    ##
######################
parameter = ParameterAntBulletA2c()

parameter = ParameterAntBulletDdpg()

parameter = ParameterAntBulletSac()

#########################
##    HOPPER_BULLET    ##
#########################
parameter = ParameterHopperBulletSac()


#########################################
##    DoubleInvertedPendulum_BULLET    ##
#########################################
parameter = ParameterDoubleInvertedPendulumBulletSac()


##########
## PONG ##
##########
parameter = ParameterPongDqn()

parameter = ParameterPongDoubleDqn()

parameter = ParameterPongDuelingDqn()

parameter = ParameterPongDoubleDuelingDqn()

parameter = ParameterPongA2c()


##################
### ANT_MUJOCO ###
##################
parameter = ParameterAntMujocoSac()


#####################
### HOPPER_MUJOCO ###
#####################
parameter = ParameterHopperMujocoSac()

#######################
### WALKER2d_MUJOCO ###
#######################
parameter = ParameterWalker2dMujocoSac()

##########################
### HALFCHEETAH_MUJOCO ###
##########################
parameter = ParameterHalfCheetahMujocoSac()

##########################
### Unity3DBall ###
##########################
parameter = Parameter3DBallDdqg()


##########################
### UnityWalker ###
##########################
parameter = ParameterWalkerDdqg()


###########################################################


if __name__ == "__main__":
    parameter = ParameterCartPoleDqn()
    print_basic_info(parameter=parameter)
