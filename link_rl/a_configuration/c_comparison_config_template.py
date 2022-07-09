#################
#  FROZEN_LAKE  #
#################
from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_frozen_lake import \
    ConfigComparisonFrozenLakeDqnActionMasking
config_c = ConfigComparisonFrozenLakeDqnActionMasking()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_frozen_lake import \
    ConfigComparisonFrozenLakeDqnTime
config_c = ConfigComparisonFrozenLakeDqnTime()


##################
#  MOUNTAIN_CAR  #
##################
from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_mountain_car import \
    ConfigComparisonMountainCarDoubleDqnRecurrentWithoutVelocity
config_c = ConfigComparisonMountainCarDoubleDqnRecurrentWithoutVelocity()


###############
#  CART_POLE  #
###############
from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import \
    ConfigComparisonCartPoleDqn
config_c = ConfigComparisonCartPoleDqn()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import \
    ConfigComparisonCartPoleDqnTypes
config_c = ConfigComparisonCartPoleDqnTypes()
config_c.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/ConfigComparisonCartPoleDqnTypes_Comparison_yhhan/reports/Comparison_CartPole_Dqn_Types--VmlldzoxODIzNDM1?accessToken=clgmrpy3zegf7b634f2cggnc3umylvvol0iujb9etabqya00r3cfrn9ekc816a89"

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import ConfigComparisonCartPoleDqnPer
config_c = ConfigComparisonCartPoleDqnPer()
config_c.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/ConfigComparisonCartPoleDqnPer_Comparison_yhhan/reports/Comparison_CartPole_Dqn_Per_Comparison_yhhan--VmlldzoxODMyNzA0?accessToken=02ju0016483twfnr44956n5muo09c6wi6yj2u9296gj3lhb43hu7c2vfbvofn7j2"

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import \
    ConfigComparisonCartPolePpo
config_c = ConfigComparisonCartPolePpo()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import \
    ConfigComparisonCartPoleDqnRecurrent
config_c = ConfigComparisonCartPoleDqnRecurrent()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import \
    ConfigComparisonCartPoleDqnRecurrentReversAction
config_c = ConfigComparisonCartPoleDqnRecurrentReversAction()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import \
    ConfigComparisonCartPoleDoubleDqnRecurrent
config_c = ConfigComparisonCartPoleDoubleDqnRecurrent()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_cart_pole import \
    ConfigComparisonCartPoleDoubleDqnRecurrentWithoutVelocity
config_c = ConfigComparisonCartPoleDoubleDqnRecurrentWithoutVelocity()


##########
#  PONG  #
##########
from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_pong import \
    ConfigComparisonPongDqn
config_c = ConfigComparisonPongDqn()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_pong import \
    ConfigComparisonPongDqnTypes
config_c = ConfigComparisonPongDqnTypes()


##################
#  LUNAR_LANDER  #
##################
from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_lunar_lander import \
    ConfigComparisonLunarLanderDqnTypes
config_c = ConfigComparisonLunarLanderDqnTypes()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_lunar_lander import \
    ConfigComparisonLunarLanderDqnRecurrent
config_c = ConfigComparisonLunarLanderDqnRecurrent()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_lunar_lander import \
    ConfigComparisonLunarLanderDqnRecurrentWithoutVelocity
config_c = ConfigComparisonLunarLanderDqnRecurrentWithoutVelocity()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_lunar_lander import \
    ConfigComparisonLunarLanderDoubleDqnRecurrent
config_c = ConfigComparisonLunarLanderDoubleDqnRecurrent()

from link_rl.a_configuration.c_comparison_config.open_ai_gym.config_comparison_lunar_lander import \
    ConfigComparisonLunarLanderDoubleDqnRecurrentWithoutVelocity
config_c = ConfigComparisonLunarLanderDoubleDqnRecurrentWithoutVelocity()


###################
#  HOPPER_MUJOCO  #
###################
from link_rl.a_configuration.c_comparison_config.mujoco.config_comparison_hopper import \
    ConfigComparisonHopperMujocoSac
config_c = ConfigComparisonHopperMujocoSac()

from link_rl.a_configuration.c_comparison_config.mujoco.config_comparison_hopper import \
    ConfigComparisonHopperMujocoSacPpo
config_c = ConfigComparisonHopperMujocoSacPpo()


#####################
#  WALKER2d_MUJOCO  #
#####################
from link_rl.a_configuration.c_comparison_config.mujoco.config_comparison_walker2d import \
    ConfigComparisonWalker2dMujocoSac
config_c = ConfigComparisonWalker2dMujocoSac()

from link_rl.a_configuration.c_comparison_config.mujoco.config_comparison_walker2d import \
    ConfigComparisonWalker2dMujocoSacPpo
config_c = ConfigComparisonWalker2dMujocoSacPpo()


########################
#  HALFCHEETAH_MUJOCO  #
########################
from link_rl.a_configuration.c_comparison_config.mujoco.config_comparison_halfcheetah import \
    ConfigComparisonHalfCheetahMujocoSac
config_c = ConfigComparisonHalfCheetahMujocoSac()

from link_rl.a_configuration.c_comparison_config.mujoco.config_comparison_halfcheetah import \
    ConfigComparisonHalfCheetahMujocoSacPpo
config_c = ConfigComparisonHalfCheetahMujocoSacPpo()


################
#  ANT_MUJOCO  #
################
from link_rl.a_configuration.c_comparison_config.mujoco.config_comparison_ant import \
    ConfigComparisonAntMujocoSac
config_c = ConfigComparisonAntMujocoSac()

from link_rl.a_configuration.c_comparison_config.mujoco.config_comparison_ant import \
    ConfigComparisonAntMujocoSacPpo
config_c = ConfigComparisonAntMujocoSacPpo()


####################
#  TaskAllocation  #
####################
from link_rl.a_configuration.c_comparison_config.combinatorial_optimization.config_comparison_task_allocation import \
    ConfigComparisonTaskAllocationDqnTypes
config_c = ConfigComparisonTaskAllocationDqnTypes()


##############
#  Knapsack  #
##############
from link_rl.a_configuration.c_comparison_config.combinatorial_optimization.config_comparison_knapsack import \
    ConfigComparisonKnapsack0StaticTestLinearDqnA2cPpo
config_c = ConfigComparisonKnapsack0StaticTestLinearDqnA2cPpo()

from link_rl.a_configuration.c_comparison_config.combinatorial_optimization.config_comparison_knapsack import \
    ConfigComparisonKnapsack0StaticTestLinearDqn
config_c = ConfigComparisonKnapsack0StaticTestLinearDqn()
config_c.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/ConfigComparisonKnapsack0StaticTestLinearDqn_Comparison_yhhan/reports/Knapsack0StaticTestLinearDqn_Comparison--VmlldzoyMDI0MzEz?accessToken=3a41c3eqjby8x7tjg7j3micxkzlz9iobjy12ld98j797uowgte9kdf3j33biyojx"

from link_rl.a_configuration.c_comparison_config.combinatorial_optimization.config_comparison_knapsack import \
    ConfigComparisonKnapsack0StaticTestLinearRecurrentDqn
config_c = ConfigComparisonKnapsack0StaticTestLinearRecurrentDqn()

from link_rl.a_configuration.c_comparison_config.combinatorial_optimization.config_comparison_knapsack import \
    ConfigComparisonKnapsack0StaticTestLinearDoubleDqn
config_c = ConfigComparisonKnapsack0StaticTestLinearDoubleDqn()

from link_rl.a_configuration.c_comparison_config.combinatorial_optimization.config_comparison_knapsack import \
    ConfigComparisonKnapsack0RandomTestLinearDoubleDqn
config_c = ConfigComparisonKnapsack0RandomTestLinearDoubleDqn()

from link_rl.a_configuration.c_comparison_config.combinatorial_optimization.config_comparison_knapsack import \
    ConfigComparisonKnapsack0RandomTestLinearDoubleDqnHer
config_c = ConfigComparisonKnapsack0RandomTestLinearDoubleDqnHer()


if __name__ == "__main__":
    from link_rl.g_utils.commons import get_env_info, print_comparison_basic_info, set_config
    from link_rl.g_utils.types import AgentType

    for config in config_c.AGENT_PARAMETERS:
        assert config.AGENT_TYPE not in (AgentType.REINFORCE,)
        set_config(config)

    observation_space, action_space = get_env_info(config_c)
    print_comparison_basic_info(observation_space, action_space, config_c)
