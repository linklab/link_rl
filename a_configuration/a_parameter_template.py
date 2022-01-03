from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleDqn, ParameterCartPoleDoubleDqn, \
    ParameterCartPoleDuelingDqn
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleReinforce
from a_configuration.c_parameters.open_ai_gym.parameter_cartpole import ParameterCartPoleA2c
from a_configuration.c_parameters.open_ai_gym.parameter_lunar_lander import ParameterLunarLanderContinuousA2c, \
    ParameterLunarLanderA2c, ParameterLunarLanderContinuousDdpg, ParameterLunarLanderContinuousSac
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongDqn
from a_configuration.c_parameters.open_ai_gym.parameter_pong import ParameterPongA2c

from a_configuration.c_parameters.pybullet.parameter_cartpole_bullet import ParameterCartPoleBulletDqn, \
    ParameterCartPoleContinuousBulletA2c, ParameterCartPoleContinuousBulletSac, ParameterCartPoleContinuousBulletDdpg
from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletA2c, ParameterAntBulletSac
from a_configuration.c_parameters.pybullet.parameter_ant_bullet import ParameterAntBulletDdpg
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
    parameter = parameter_cart_pole_duelingdqn
    print_basic_info(parameter=parameter)
