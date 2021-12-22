from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleReinforce
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleA2c
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDqn
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongA2c

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletDqn, \
    ParameterCartPoleContinuousBulletA2c
from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletA2c
from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletDdpg
from g_utils.commons import print_basic_info

parameter_list = []

###############
## CART_POLE ##
###############
parameter_cart_pole_dqn = ParameterCartPoleDqn()
parameter_list.append(parameter_cart_pole_dqn)

parameter_cart_pole_reinforce = ParameterCartPoleReinforce()
parameter_list.append(parameter_cart_pole_reinforce)

parameter_cart_pole_a2c = ParameterCartPoleA2c()
parameter_list.append(parameter_cart_pole_a2c)


######################
## CART_POLE_BULLET ##
######################
parameter_cart_pole_bullet_dqn = ParameterCartPoleBulletDqn()
parameter_list.append(parameter_cart_pole_bullet_dqn)

parameter_cart_pole_continuous_bullet_a2c = ParameterCartPoleContinuousBulletA2c()
parameter_list.append(parameter_cart_pole_continuous_bullet_a2c)


######################
##    ANT_BULLET    ##
######################
parameter_ant_bullet_a2c = ParameterAntBulletA2c()
parameter_list.append(parameter_ant_bullet_a2c)

parameter_ant_bullet_ddpg = ParameterAntBulletDdpg()
parameter_list.append(parameter_ant_bullet_ddpg)


##########
## PONG ##
##########
parameter_pong_dqn = ParameterPongDqn()
parameter_list.append(parameter_pong_dqn)

parameter_pong_a2c = ParameterPongA2c()
parameter_list.append(parameter_pong_a2c)


###########################################################
for parameter in parameter_list:
    parameter.USE_WANDB = False
    parameter.WANDB_ENTITY = "link-koreatech"


if __name__ == "__main__":
    parameter = parameter_pong_a2c
    print_basic_info(device=None, parameter=parameter)
