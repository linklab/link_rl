import gym
import numpy as np

from link_rl.b_environments import wrapper
from link_rl.d_models.b_q_model import Q_MODEL

#################
## FROZEN_LAKE ##
#################
from link_rl.a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn
config = ConfigFrozenLakeDqn()
config.ACTION_MASKING = True
config.RANDOM_MAP = False

from link_rl.a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn
config = ConfigFrozenLakeDqn()
config.ENV_KWARGS["is_slippery"] = False
config.ENV_KWARGS["desc"] = ["SFF", "FHF", "FFG"]
config.WRAPPERS.append(
    (gym.wrappers.TransformObservation, {"f": lambda obs: np.random.randn(*obs.shape)})
)

from link_rl.a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn
config = ConfigFrozenLakeDqn()
config.ENV_KWARGS["is_slippery"] = False
config.ENV_KWARGS["desc"] = ["SFF", "FHF", "FFG"]
config.WRAPPERS.append((gym.wrappers.TimeAwareObservation, dict()))
config.WRAPPERS.append(
    (gym.wrappers.TransformObservation, {"f": lambda obs: np.random.randn(*obs.shape)})
)
config.WRAPPERS.append(
    (gym.wrappers.TimeAwareObservation, {})
)

from link_rl.a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn
config = ConfigFrozenLakeDqn()
config.ENV_KWARGS["is_slippery"] = False
config.ENV_KWARGS["desc"] = ["SFF", "FHF", "FFG"]
config.MODEL_TYPE = Q_MODEL.QModel.value
config.WRAPPERS.append(
    (gym.wrappers.TransformObservation, {"f": lambda obs: np.random.randn(*obs.shape)})
)

###############
## CART_POLE ##
###############
from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/CartPole-v1_Dqn_anonymous/reports/Cartpole-DQN--VmlldzoxNzI3NzU0?accessToken=5m3zzlj30hpflat7mb9zu3ygql9ani0lox22y5ermhocotbakar4so5lq8pe86gk"

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()
config.WRAPPERS.append(
    (wrapper.ReverseActionCartpole, {})
)

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()
config.WRAPPERS.append(
    (wrapper.ReverseActionCartpole, {})
)
config.WRAPPERS.append(
    (gym.wrappers.TimeAwareObservation, {})
)

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()
# config.WRAPPERS.append(
#     (wrapper.ReverseActionCartpole, {})
# )
config.MODEL_TYPE = Q_MODEL.QModel.value

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDoubleDqn
config = ConfigCartPoleDoubleDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDuelingDqn
config = ConfigCartPoleDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDoubleDuelingDqn
config = ConfigCartPoleDoubleDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleReinforce
config = ConfigCartPoleReinforce()

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleA2c
config = ConfigCartPoleA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleA3c
config = ConfigCartPoleA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPolePpo
config = ConfigCartPolePpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPolePpoTrajectory
config = ConfigCartPolePpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/CartPole-v1_PPO_TRAJECTORY_yhhan/reports/CartPole-PPO_TRAJECTORY--VmlldzoxNTUyNDQw?accessToken=7vs10v66vi6fyitrxma0p2g62pq4nfiiccaskfgv2x2b5jrmwfae38u4pfm1xfq1"

##################
## LUNAR_LANDER ##
##################
from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderDqn
config = ConfigLunarLanderDqn()
config.WANDB_REPORt_URL = 'https://wandb.ai/link-koreatech/LunarLander-v2_DQN_anonymous/reports/Lunar-Lander-DQN--VmlldzoxNzI3OTU5?accessToken=4kbmznjvysvlh5zbf30vy8jg403ojrjxjefiewfpcn5iu3cwo5mirclsj8f6dx1u'

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderA2c
config = ConfigLunarLanderA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderA3c
config = ConfigLunarLanderA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderPpo
config = ConfigLunarLanderPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderPpoTrajectory
config = ConfigLunarLanderPpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLander-v2_PPO_TRAJECTORY_anonymous/reports/LunarLander-v2-PPO_TRAJECTORY--VmlldzoxNTAwOTg0?accessToken=y2tgu6i38zzehklrq6rikdqhvza2fsexmfg22ntl2drl0q9bsvd11t1b0e09r0ry"

#############################
## LUNAR_LANDER_CONTINUOUS ##
#############################
from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousA2c
config = ConfigLunarLanderContinuousA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousA3c
config = ConfigLunarLanderContinuousA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousPpo
config = ConfigLunarLanderContinuousPpo()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_PPO_yhhan/reports/LunarLanderContinuous-v2_PPO--VmlldzoxOTkyMDUy?accessToken=von6k6x7zlz1fns9gb7ld7lqpsvtemc3z6yx70t43iuxarppga7v0f0ard4uf18t"

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousAsynchronousPpo
config = ConfigLunarLanderContinuousAsynchronousPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousPpoTrajectory
config = ConfigLunarLanderContinuousPpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_PPO_TRAJECTORY_yhhan/reports/LunarLanderContinuous-v2_PPO_TRAJECTORY--VmlldzoxNTA3MjAx"

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousDdpg
config = ConfigLunarLanderContinuousDdpg()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousTd3
config = ConfigLunarLanderContinuousTd3()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_TD3_yhhan/reports/LunarLanderContinuous-v2_TD3--VmlldzoxNTA3MjA4"

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousSac
config = ConfigLunarLanderContinuousSac()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_SAC_yhhan/reports/LunarLanderContinuous-v2_SAC--VmlldzoxNTA3MjA2"

###########################
## Normal Bipedal Walker ##
###########################
from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerA2c
config = ConfigNormalBipedalWalkerA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerA3c
config = ConfigNormalBipedalWalkerA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerPpo
config = ConfigNormalBipedalWalkerPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerPpoTrajectory
config = ConfigNormalBipedalWalkerPpoTrajectory()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerDdpg
config = ConfigNormalBipedalWalkerDdpg()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerTd3
config = ConfigNormalBipedalWalkerTd3()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerSac
config = ConfigNormalBipedalWalkerSac()

#############################
## Hardcore Bipedal Walker ##
#############################
from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerA2c
config = ConfigHardcoreBipedalWalkerA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerA3c
config = ConfigHardcoreBipedalWalkerA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerPpo
config = ConfigHardcoreBipedalWalkerPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerPpoTrajectory
config = ConfigHardcoreBipedalWalkerPpoTrajectory()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerDdpg
config = ConfigHardcoreBipedalWalkerDdpg()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerTd3
config = ConfigHardcoreBipedalWalkerTd3()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerSac
config = ConfigHardcoreBipedalWalkerSac()


################
## Car Racing ##
################
from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingA2c
config = ConfigCarRacingA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingA3c
config = ConfigCarRacingA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingPpo
config = ConfigCarRacingPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingPpoTrajectory
config = ConfigCarRacingPpoTrajectory()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingDdpg
config = ConfigCarRacingDdpg()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingTd3
config = ConfigCarRacingTd3()

from link_rl.a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingSac
config = ConfigCarRacingSac()

#########################################
##    InvertedDoublePendulumMujoco    ##
#########################################
from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_inverted_double_pendulum_mujoco import ConfigInvertedDoublePendulumMujocoPpo
config = ConfigInvertedDoublePendulumMujocoPpo()

##########
## PONG ##
##########
from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDqn
config = ConfigPongDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDoubleDqn
config = ConfigPongDoubleDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDuelingDqn
config = ConfigPongDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDoubleDuelingDqn
config = ConfigPongDoubleDuelingDqn()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/Pong-v5_DOUBLE_DUELING_DQN_yhhan/reports/Pong-v5_DOUBLE_DUELING_DQN--VmlldzoxODYxMjU4?accessToken=gsvpknqu1gqi6cb0177jgdzk61oqhpua4kcj680l89fdvwer1cg3wms7rb62syj8"

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongA2c
config = ConfigPongA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongA3c
config = ConfigPongA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongPpo
config = ConfigPongPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongPpoTrajectory
config = ConfigPongPpoTrajectory()

##########
## Breakout ##
##########
from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutDqn
config = ConfigBreakoutDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutDoubleDqn
config = ConfigBreakoutDoubleDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutDuelingDqn
config = ConfigBreakoutDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutDoubleDuelingDqn
config = ConfigBreakoutDoubleDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutA2c
config = ConfigBreakoutA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutA3c
config = ConfigBreakoutA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutPpo
config = ConfigBreakoutPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutPpoTrajectory
config = ConfigBreakoutPpoTrajectory()

##########
## VideoPinball ##
##########
from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballDqn
config = ConfigVideoPinballDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballDoubleDqn
config = ConfigVideoPinballDoubleDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballDuelingDqn
config = ConfigVideoPinballDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballDoubleDuelingDqn
config = ConfigVideoPinballDoubleDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballA2c
config = ConfigVideoPinballA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballA3c
config = ConfigVideoPinballA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballPpo
config = ConfigVideoPinballPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballPpoTrajectory
config = ConfigVideoPinballPpoTrajectory()


#################
## DemonAttack ##
#################
from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_demon_attack import ConfigDemonAttackDqn
config = ConfigDemonAttackDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_demon_attack import ConfigDemonAttackDoubleDqn
config = ConfigDemonAttackDoubleDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_demon_attack import ConfigDemonAttackDuelingDqn
config = ConfigDemonAttackDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_demon_attack import ConfigDemonAttackDoubleDuelingDqn
config = ConfigDemonAttackDoubleDuelingDqn()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_demon_attack import ConfigDemonAttackA2c
config = ConfigDemonAttackA2c()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_demon_attack import ConfigDemonAttackA3c
config = ConfigDemonAttackA3c()

from link_rl.a_configuration.b_single_config.open_ai_gym.atari.config_demon_attack import ConfigDemonAttackPpo
config = ConfigDemonAttackPpo()

##################
### ANT_MUJOCO ###
##################
from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_ant_mujoco import ConfigAntMujocoSac
config = ConfigAntMujocoSac()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/Ant-v2_SAC_yhhan/reports/Ant-v2_SAC--VmlldzoxNTM0NDMz"

from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_ant_mujoco import ConfigAntMujocoPpo
config = ConfigAntMujocoPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_ant_mujoco import ConfigAntMujocoPpoTrajectory
config = ConfigAntMujocoPpoTrajectory()

#####################
### HOPPER_MUJOCO ###
#####################
from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_hopper_mujoco import ConfigHopperMujocoSac
config = ConfigHopperMujocoSac()

from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_hopper_mujoco import ConfigHopperMujocoPpo
config = ConfigHopperMujocoPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_hopper_mujoco import ConfigHopperMujocoPpoTrajectory
config = ConfigHopperMujocoPpoTrajectory()

#######################
### WALKER2d_MUJOCO ###
#######################
from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_walker2d_mujoco import ConfigWalker2dMujocoSac
config = ConfigWalker2dMujocoSac()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/Walker2d-v2_SAC_yhhan/reports/3DBall-SAC--VmlldzoxNTI4NDk4?accessToken=ikha9ymzoqh7o3zefs6cn0ao5hmz7qzc8wnx0shd7i63wesx8585ja901i1bie1z"

from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_walker2d_mujoco import ConfigWalker2dMujocoPpo
config = ConfigWalker2dMujocoPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_walker2d_mujoco import ConfigWalker2dMujocoPpoTrajectory
config = ConfigWalker2dMujocoPpoTrajectory()

##########################
### HALFCHEETAH_MUJOCO ###
##########################
from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoSac
config = ConfigHalfCheetahMujocoSac()

from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoPpo
config = ConfigHalfCheetahMujocoPpo()

from link_rl.a_configuration.b_single_config.open_ai_gym.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoPpoTrajectory
config = ConfigHalfCheetahMujocoPpoTrajectory()

##########################
### Unity3DBall ###
##########################
from link_rl.a_configuration.b_single_config.unity.config_3d_ball import Config3DBallDdqg
config = Config3DBallDdqg()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/Unity3DBall_DDPG_yhhan/reports/3DBall-Ddqg--VmlldzoxNTM0NTUy?accessToken=gxnxojzda8hu41wb52vv873kql0qkir16nyqfcrrzq6eeaozen9xfe2pmg4f0dt5"


##########################
### UnityWalker ###
##########################
from link_rl.a_configuration.b_single_config.unity.config_walker import ConfigWalkerSac
config = ConfigWalkerSac()


##########################
### UnityDrone ###
##########################
from link_rl.a_configuration.b_single_config.unity.config_drone import ConfigDroneDdpg
config = ConfigDroneDdpg()

from link_rl.a_configuration.b_single_config.unity.config_drone import ConfigDroneSac
config = ConfigDroneSac()

###########################
#### Knapsack - Random ####
###########################
from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0RandomTestDqn
config = ConfigKnapsack0RandomTestDqn()


##################################
#### Knapsack - Random Linear ####
##################################
from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0RandomTestLinearDqn
config = ConfigKnapsack0RandomTestLinearDqn()

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0RandomTestLinearA2c
config = ConfigKnapsack0RandomTestLinearA2c()

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0RandomTestLinearPpo
config = ConfigKnapsack0RandomTestLinearPpo()


#################################
######## Knapsack - Load ########
#################################
from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0LoadTestDqn
config = ConfigKnapsack0LoadTestDqn()


#################################
#### Knapsack - Load Linear #####
#################################
from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0LoadTestLinearDqn
config = ConfigKnapsack0LoadTestLinearDqn()

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0LoadTestLinearA2c
config = ConfigKnapsack0LoadTestLinearA2c()

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0LoadTestLinearPpo
config = ConfigKnapsack0LoadTestLinearPpo()


#################################
####### Knapsack - static #######
#################################
from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestDqn
config = ConfigKnapsack0StaticTestDqn()


########################################
####### Knapsack - Static Linear #######
########################################

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearDqn
config = ConfigKnapsack0StaticTestLinearDqn()

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearDoubleDqn
config = ConfigKnapsack0StaticTestLinearDoubleDqn()

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearDoubleDuelingDqn
config = ConfigKnapsack0StaticTestLinearDoubleDuelingDqn()

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearA2c
config = ConfigKnapsack0StaticTestLinearA2c()

from link_rl.a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearPpo
config = ConfigKnapsack0StaticTestLinearPpo()


###################
### GymRobotics ###
###################

from link_rl.a_configuration.b_single_config.gym_robotics.config_gym_robotics_hand_manipulate_block_rotate_xyz import ConfigHandManipulateBlockRotateXYZA2c
config = ConfigHandManipulateBlockRotateXYZA2c()

from link_rl.a_configuration.b_single_config.gym_robotics.config_gym_robotics_hand_manipulate_block_rotate_xyz import ConfigHandManipulateBlockRotateXYZPpo
config = ConfigHandManipulateBlockRotateXYZPpo()

from link_rl.a_configuration.b_single_config.gym_robotics.config_gym_robotics_hand_manipulate_block_rotate_xyz import ConfigHandManipulateBlockRotateXYZTd3
config = ConfigHandManipulateBlockRotateXYZTd3()

from link_rl.a_configuration.b_single_config.gym_robotics.config_gym_robotics_hand_manipulate_block_rotate_xyz import ConfigHandManipulateBlockRotateXYZSac
config = ConfigHandManipulateBlockRotateXYZSac()


####################################
### CompetitionOlympics-Football ###
####################################

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_football import ConfigCompetitionOlympicsFootballA3c
config = ConfigCompetitionOlympicsFootballA3c()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_football import ConfigCompetitionOlympicsFootballPpo
config = ConfigCompetitionOlympicsFootballPpo()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_football import ConfigCompetitionOlympicsFootballAsynchronousPpo
config = ConfigCompetitionOlympicsFootballAsynchronousPpo()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_football import ConfigCompetitionOlympicsFootballTd3
config = ConfigCompetitionOlympicsFootballTd3()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_football import ConfigCompetitionOlympicsFootballSac
config = ConfigCompetitionOlympicsFootballSac()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_football import ConfigCompetitionOlympicsFootballTdmpc
config = ConfigCompetitionOlympicsFootballTdmpc()


#####################################
### CompetitionOlympics-Wrestling ###
#####################################

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_wrestling import ConfigCompetitionOlympicsWrestlingA3c
config = ConfigCompetitionOlympicsWrestlingA3c()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_wrestling import ConfigCompetitionOlympicsWrestlingPpo
config = ConfigCompetitionOlympicsWrestlingPpo()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_wrestling import ConfigCompetitionOlympicsWrestlingAsynchronousPpo
config = ConfigCompetitionOlympicsWrestlingAsynchronousPpo()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_wrestling import ConfigCompetitionOlympicsWrestlingTd3
config = ConfigCompetitionOlympicsWrestlingTd3()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_wrestling import ConfigCompetitionOlympicsWrestlingSac
config = ConfigCompetitionOlympicsWrestlingSac()


####################################
### CompetitionOlympics-Running ###
####################################

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_running import ConfigCompetitionOlympicsRunningA3c
config = ConfigCompetitionOlympicsRunningA3c()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_running import ConfigCompetitionOlympicsRunningPpo
config = ConfigCompetitionOlympicsRunningPpo()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_running import ConfigCompetitionOlympicsRunningAsynchronousPpo
config = ConfigCompetitionOlympicsRunningAsynchronousPpo()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_running import ConfigCompetitionOlympicsRunningTd3
config = ConfigCompetitionOlympicsRunningTd3()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_running import ConfigCompetitionOlympicsRunningSac
config = ConfigCompetitionOlympicsRunningSac()


####################################
### CompetitionOlympics-TableHockey ###
####################################

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_tablehockey import ConfigCompetitionOlympicsTableHockeyA3c
config = ConfigCompetitionOlympicsTableHockeyA3c()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_tablehockey import ConfigCompetitionOlympicsTableHockeyPpo
config = ConfigCompetitionOlympicsTableHockeyPpo()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_tablehockey import ConfigCompetitionOlympicsTableHockeyAsynchronousPpo
config = ConfigCompetitionOlympicsTableHockeyAsynchronousPpo()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_tablehockey import ConfigCompetitionOlympicsTableHockeyTd3
config = ConfigCompetitionOlympicsTableHockeyTd3()

from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_integrated import ConfigCompetitionOlympicsIntegratedSac
config = ConfigCompetitionOlympicsIntegratedSac()


###########################
### BasicTaskAllocation ###
###########################
from link_rl.a_configuration.b_single_config.task_allocation.config_basic_task_allocation import ConfigBasicTaskAllocation0Dqn
config = ConfigBasicTaskAllocation0Dqn()

from link_rl.a_configuration.b_single_config.task_allocation.config_basic_task_allocation import ConfigBasicTaskAllocation1Dqn
config = ConfigBasicTaskAllocation1Dqn()


###############################
####### Task Allocation #######
###############################
from link_rl.a_configuration.b_single_config.task_allocation.config_task_allocation import ConfigTaskAllocationDqn
config = ConfigTaskAllocationDqn()

###############################
####### AI Birds #######
###############################
from link_rl.a_configuration.b_single_config.ai_birds.config_ai_birds import ConfigAiBirdsDqn
config = ConfigAiBirdsDqn()

from link_rl.a_configuration.b_single_config.ai_birds.config_ai_birds import ConfigAiBirdsDoubleDqn
config = ConfigAiBirdsDoubleDqn()

from link_rl.a_configuration.b_single_config.ai_birds.config_ai_birds import ConfigAiBirdsDuelingDqn
config = ConfigAiBirdsDuelingDqn()

from link_rl.a_configuration.b_single_config.ai_birds.config_ai_birds import ConfigAiBirdsDoubleDuelingDqn
config = ConfigAiBirdsDoubleDuelingDqn()

from link_rl.a_configuration.b_single_config.ai_birds.config_ai_birds import ConfigAiBirdsA2c
config = ConfigAiBirdsA2c()

from link_rl.a_configuration.b_single_config.ai_birds.config_ai_birds import ConfigAiBirdsPpo
config = ConfigAiBirdsPpo()

###############################
######## Evolution Gym ########
###############################

# Climber
from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_climber import ConfigEvolutionGymClimberV0Sac
config = ConfigEvolutionGymClimberV0Sac()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_climber import ConfigEvolutionGymClimberV1Sac
config = ConfigEvolutionGymClimberV1Sac()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_climber import ConfigEvolutionGymClimberV2Sac
config = ConfigEvolutionGymClimberV2Sac()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_climber import ConfigEvolutionGymClimberV0Tdmpc
config = ConfigEvolutionGymClimberV0Tdmpc()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_climber import ConfigEvolutionGymClimberV1Tdmpc
config = ConfigEvolutionGymClimberV1Tdmpc()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_climber import ConfigEvolutionGymClimberV2Tdmpc
config = ConfigEvolutionGymClimberV2Tdmpc()


# Walker

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_walker import ConfigEvolutionGymWalkerSac
config = ConfigEvolutionGymWalkerSac()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_walker import ConfigEvolutionGymBridgeWalkerSac
config = ConfigEvolutionGymBridgeWalkerSac()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_walker import ConfigEvolutionGymCaveCrawlerSac
config = ConfigEvolutionGymCaveCrawlerSac()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_walker import ConfigEvolutionGymWalkerTdmpc
config = ConfigEvolutionGymWalkerTdmpc()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_walker import ConfigEvolutionGymBridgeWalkerTdmpc
config = ConfigEvolutionGymBridgeWalkerTdmpc()

from link_rl.a_configuration.b_single_config.evolution_gym.config_evolution_gym_walker import ConfigEvolutionGymCaveCrawlerTdmpc
config = ConfigEvolutionGymCaveCrawlerTdmpc()


if __name__ == "__main__":
    from link_rl.h_utils.commons import print_basic_info, get_env_info, set_config

    set_config(config)
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)