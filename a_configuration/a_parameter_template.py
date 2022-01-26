###############
## CART_POLE ##
###############
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn
parameter = ParameterCartPoleDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDoubleDqn
parameter = ParameterCartPoleDoubleDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDuelingDqn
parameter = ParameterCartPoleDuelingDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDoubleDuelingDqn
parameter = ParameterCartPoleDoubleDuelingDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleReinforce
parameter = ParameterCartPoleReinforce()

from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleA2c
parameter = ParameterCartPoleA2c()

from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPolePpo
parameter = ParameterCartPolePpo()

##################
## LUNAR_LANDER ##
##################
from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderDqn
parameter = ParameterLunarLanderDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderA2c
parameter = ParameterLunarLanderA2c()

from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderPpo
parameter = ParameterLunarLanderPpo()

from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderContinuousA2c
parameter = ParameterLunarLanderContinuousA2c()

from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderContinuousPpo
parameter = ParameterLunarLanderContinuousPpo()

from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderContinuousDdpg
parameter = ParameterLunarLanderContinuousDdpg()

from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderContinuousSac
parameter = ParameterLunarLanderContinuousSac()

######################
## CART_POLE_BULLET ##
######################
from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletDqn
parameter = ParameterCartPoleBulletDqn()

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletDoubleDqn
parameter = ParameterCartPoleBulletDoubleDqn()

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletDuelingDqn
parameter = ParameterCartPoleBulletDuelingDqn()

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletDoubleDuelingDqn
parameter = ParameterCartPoleBulletDoubleDuelingDqn()

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletA2c
parameter = ParameterCartPoleBulletA2c()

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletPpo
parameter = ParameterCartPoleBulletPpo()


#################################
## CART_POLE_CONTINUOUS_BULLET ##
#################################
from a_configuration.c_parameters.pybullet.parameter_cartpole_continuous_bullet import ParameterCartPoleContinuousBulletA2c
parameter = ParameterCartPoleContinuousBulletA2c()

from a_configuration.c_parameters.pybullet.parameter_cartpole_continuous_bullet import ParameterCartPoleContinuousBulletPpo
parameter = ParameterCartPoleContinuousBulletPpo()

from a_configuration.c_parameters.pybullet.parameter_cartpole_continuous_bullet import ParameterCartPoleContinuousBulletSac
parameter = ParameterCartPoleContinuousBulletSac()

from a_configuration.c_parameters.pybullet.parameter_cartpole_continuous_bullet import ParameterCartPoleContinuousBulletDdpg
parameter = ParameterCartPoleContinuousBulletDdpg()

from a_configuration.c_parameters.pybullet.parameter_cartpole_continuous_bullet import ParameterCartPoleContinuousBulletTd3
parameter = ParameterCartPoleContinuousBulletTd3()

######################
##    ANT_BULLET    ##
######################
from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletA2c
parameter = ParameterAntBulletA2c()

from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletDdpg
parameter = ParameterAntBulletDdpg()

from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletSac
parameter = ParameterAntBulletSac()

#########################
##    HOPPER_BULLET    ##
#########################
from a_configuration.c_parameters.pybullet.parameter_hopper_bullet import ParameterHopperBulletSac
parameter = ParameterHopperBulletSac()


#########################################
##    DoubleInvertedPendulum_BULLET    ##
#########################################
from a_configuration.c_parameters.pybullet.parameter_double_inverted_pendulum_bullet import ParameterDoubleInvertedPendulumBulletSac
parameter = ParameterDoubleInvertedPendulumBulletSac()


##########
## PONG ##
##########
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDqn
parameter = ParameterPongDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDoubleDqn
parameter = ParameterPongDoubleDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDuelingDqn
parameter = ParameterPongDuelingDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDoubleDuelingDqn
parameter = ParameterPongDoubleDuelingDqn()

from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongA2c
parameter = ParameterPongA2c()


##################
### ANT_MUJOCO ###
##################
from a_configuration.c_parameters.mujoco.parameter_ant_mujoco import ParameterAntMujocoSac
parameter = ParameterAntMujocoSac()


#####################
### HOPPER_MUJOCO ###
#####################
from a_configuration.c_parameters.mujoco.parameter_hopper_mujoco import ParameterHopperMujocoSac
parameter = ParameterHopperMujocoSac()

#######################
### WALKER2d_MUJOCO ###
#######################
from a_configuration.c_parameters.mujoco.parameter_walker2d_mujoco import ParameterWalker2dMujocoSac
parameter = ParameterWalker2dMujocoSac()

##########################
### HALFCHEETAH_MUJOCO ###
##########################
from a_configuration.c_parameters.mujoco.parameter_halfcheetah_mujoco import ParameterHalfCheetahMujocoSac
parameter = ParameterHalfCheetahMujocoSac()

##########################
### Unity3DBall ###
##########################
from a_configuration.c_parameters.unity.parameter_3d_ball import Parameter3DBallDdqg
parameter = Parameter3DBallDdqg()


##########################
### UnityWalker ###
##########################
from a_configuration.c_parameters.unity.parameter_walker import ParameterWalkerDdqg
parameter = ParameterWalkerDdqg()

parameter.USE_WANDB = False

if __name__ == "__main__":
    from g_utils.commons import print_basic_info, get_env_info
    observation_space, action_space = get_env_info(parameter)
    print_basic_info(observation_space, action_space, parameter)