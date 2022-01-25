from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_ant import ParameterComparisonAntMujocoSac
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_halfcheetah import \
    ParameterComparisonHalfCheetahMujocoSac
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_hopper import \
    ParameterComparisonHopperMujocoSac
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_walker2d import \
    ParameterComparisonWalker2dMujocoSac
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_cartpole import \
    ParameterComparisonCartPoleDqn, ParameterComparisonCartPoleDqnTypes
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_lunarlander import \
    ParameterComparisonLunarLanderDqnRecurrent
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_pong import ParameterComparisonPongDqn, \
    ParameterComparisonPongDqnTypes
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_ant import ParameterComparisonAntBulletSac
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_cartpole_bullet import \
    ParameterComparisonCartPoleBulletA2c, ParameterComparisonCartPoleBulletDqnTypes, \
    ParameterComparisonCartPoleContinuousBulletDdpg
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_double_inverted_pendulum_bullet import \
    ParameterComparisonDoubleInvertedPendulumBulletSac
from g_utils.types import ModelType



###############
## CART_POLE ##
###############
parameter_c = ParameterComparisonCartPoleDqn()
parameter_c = ParameterComparisonCartPoleDqnTypes()

######################
## CART_POLE_BULLET ##
######################
parameter_c = ParameterComparisonCartPoleBulletA2c()
parameter_c = ParameterComparisonCartPoleBulletDqnTypes()
parameter_c = ParameterComparisonCartPoleContinuousBulletDdpg()


######################
## PONG ##
######################
parameter_c = ParameterComparisonPongDqn()
parameter_c = ParameterComparisonPongDqnTypes()


##################
## LUNAR_LANDER ##
##################
parameter_c = ParameterComparisonLunarLanderDqnRecurrent()


#####################
### HOPPER_MUJOCO ###
#####################
parameter_c = ParameterComparisonHopperMujocoSac()


#######################
### WALKER2d_MUJOCO ###
#######################
parameter_c = ParameterComparisonWalker2dMujocoSac()


##########################
### HALFCHEETAH_MUJOCO ###
##########################
parameter_c = ParameterComparisonHalfCheetahMujocoSac()


##################
### ANT_MUJOCO ###
##################
parameter_c = ParameterComparisonAntMujocoSac()


######################
##    ANT_BULLET    ##
######################
parameter_c = ParameterComparisonAntBulletSac()


#########################################
##    DoubleInvertedPendulum_BULLET    ##
#########################################
parameter_c = ParameterComparisonDoubleInvertedPendulumBulletSac()

#######################################################################
for parameter_comparison in parameter_c:
    parameter_comparison.USE_WANDB = False
    parameter_comparison.WANDB_ENTITY = "link-koreatech"

