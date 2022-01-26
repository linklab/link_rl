###############
## CART_POLE ##
###############
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_cart_pole import ParameterComparisonCartPoleDqn
from g_utils.commons import print_basic_info, get_env_info, print_comparison_basic_info

parameter_c = ParameterComparisonCartPoleDqn()

from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_cart_pole import ParameterComparisonCartPoleDqnTypes
parameter_c = ParameterComparisonCartPoleDqnTypes()

######################
## CART_POLE_BULLET ##
######################
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_cartpole_bullet import ParameterComparisonCartPoleBulletA2c
parameter_c = ParameterComparisonCartPoleBulletA2c()

from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_cartpole_bullet import ParameterComparisonCartPoleBulletDqnTypes
parameter_c = ParameterComparisonCartPoleBulletDqnTypes()

from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_cartpole_bullet import ParameterComparisonCartPoleContinuousBulletDdpg
parameter_c = ParameterComparisonCartPoleContinuousBulletDdpg()


######################
## PONG ##
######################
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_pong import ParameterComparisonPongDqn
parameter_c = ParameterComparisonPongDqn()

from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_pong import ParameterComparisonPongDqnTypes
parameter_c = ParameterComparisonPongDqnTypes()


##################
## LUNAR_LANDER ##
##################
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_lunar_lander import ParameterComparisonLunarLanderDqnRecurrent
parameter_c = ParameterComparisonLunarLanderDqnRecurrent()


#####################
### HOPPER_MUJOCO ###
#####################
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_hopper import ParameterComparisonHopperMujocoSac
parameter_c = ParameterComparisonHopperMujocoSac()


#######################
### WALKER2d_MUJOCO ###
#######################
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_walker2d import ParameterComparisonWalker2dMujocoSac
parameter_c = ParameterComparisonWalker2dMujocoSac()


##########################
### HALFCHEETAH_MUJOCO ###
##########################
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_halfcheetah import ParameterComparisonHalfCheetahMujocoSac
parameter_c = ParameterComparisonHalfCheetahMujocoSac()


##################
### ANT_MUJOCO ###
##################
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_ant import ParameterComparisonAntMujocoSac
parameter_c = ParameterComparisonAntMujocoSac()


######################
##    ANT_BULLET    ##
######################
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_ant import ParameterComparisonAntBulletSac
parameter_c = ParameterComparisonAntBulletSac()


#########################################
##    DoubleInvertedPendulum_BULLET    ##
#########################################
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_double_inverted_pendulum_bullet import ParameterComparisonDoubleInvertedPendulumBulletSac
parameter_c = ParameterComparisonDoubleInvertedPendulumBulletSac()

parameter_c.USE_WANDB = False

if __name__ == "__main__":
    observation_space, action_space = get_env_info(parameter_c)
    print_comparison_basic_info(observation_space, action_space, parameter_c)

