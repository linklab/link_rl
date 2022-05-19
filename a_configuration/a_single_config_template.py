import gym
import numpy as np

from b_environments import wrapper
from g_utils.types import ModelType

#################
## FROZEN_LAKE ##
#################
from a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn
config = ConfigFrozenLakeDqn()
config.ACTION_MASKING = True
config.RANDOM_MAP = False

from a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn
config = ConfigFrozenLakeDqn()
config.ENV_KWARGS["is_slippery"] = False
config.ENV_KWARGS["desc"] = ["SFF", "FHF", "FFG"]
config.WRAPPERS.append(
    (gym.wrappers.TransformObservation, {"f": lambda obs: np.random.randn(*obs.shape)})
)

from a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn
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

from a_configuration.b_single_config.open_ai_gym.toy_text.config_frozen_lake import ConfigFrozenLakeDqn
config = ConfigFrozenLakeDqn()
config.ENV_KWARGS["is_slippery"] = False
config.ENV_KWARGS["desc"] = ["SFF", "FHF", "FFG"]
config.MODEL_TYPE = ModelType.TINY_RECURRENT
config.WRAPPERS.append(
    (gym.wrappers.TransformObservation, {"f": lambda obs: np.random.randn(*obs.shape)})
)

###############
## CART_POLE ##
###############
from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/CartPole-v1_Dqn_anonymous/reports/Cartpole-DQN--VmlldzoxNzI3NzU0?accessToken=5m3zzlj30hpflat7mb9zu3ygql9ani0lox22y5ermhocotbakar4so5lq8pe86gk"

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()
config.WRAPPERS.append(
    (wrapper.ReverseActionCartpole, {})
)

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()
config.WRAPPERS.append(
    (wrapper.ReverseActionCartpole, {})
)
config.WRAPPERS.append(
    (gym.wrappers.TimeAwareObservation, {})
)

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDqn
config = ConfigCartPoleDqn()
config.WRAPPERS.append(
    (wrapper.ReverseActionCartpole, {})
)
config.MODEL_TYPE = ModelType.SMALL_RECURRENT

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDoubleDqn
config = ConfigCartPoleDoubleDqn()

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDuelingDqn
config = ConfigCartPoleDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleDoubleDuelingDqn
config = ConfigCartPoleDoubleDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleReinforce
config = ConfigCartPoleReinforce()

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleA2c
config = ConfigCartPoleA2c()

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleA3c
config = ConfigCartPoleA3c()

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPolePpo
config = ConfigCartPolePpo()

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPolePpoTrajectory
config = ConfigCartPolePpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/CartPole-v1_PPO_TRAJECTORY_yhhan/reports/CartPole-PPO_TRAJECTORY--VmlldzoxNTUyNDQw?accessToken=7vs10v66vi6fyitrxma0p2g62pq4nfiiccaskfgv2x2b5jrmwfae38u4pfm1xfq1"

from a_configuration.b_single_config.open_ai_gym.classic_control.config_cart_pole import ConfigCartPoleMuzero
config = ConfigCartPoleMuzero()

######################
## CART_POLE_BULLET ##
######################
from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletDqn
config = ConfigCartPoleBulletDqn()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletDoubleDqn
config = ConfigCartPoleBulletDoubleDqn()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletDuelingDqn
config = ConfigCartPoleBulletDuelingDqn()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletDoubleDuelingDqn
config = ConfigCartPoleBulletDoubleDuelingDqn()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletA2c
config = ConfigCartPoleBulletA2c()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletA3c
config = ConfigCartPoleBulletA3c()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletPpo
config = ConfigCartPoleBulletPpo()

from a_configuration.b_single_config.pybullet.config_cart_pole_bullet import ConfigCartPoleBulletPpoTrajectory
config = ConfigCartPoleBulletPpoTrajectory()

#################################
## CART_POLE_CONTINUOUS_BULLET ##
#################################
from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletReinforce
config = ConfigCartPoleContinuousBulletReinforce()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletA2c
config = ConfigCartPoleContinuousBulletA2c()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletA3c
config = ConfigCartPoleContinuousBulletA3c()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletPpo
config = ConfigCartPoleContinuousBulletPpo()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletPpoTrajectory
config = ConfigCartPoleContinuousBulletPpoTrajectory()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletDdpg
config = ConfigCartPoleContinuousBulletDdpg()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletTd3
config = ConfigCartPoleContinuousBulletTd3()

from a_configuration.b_single_config.pybullet.config_cart_pole_continuous_bullet import ConfigCartPoleContinuousBulletSac
config = ConfigCartPoleContinuousBulletSac()

##################
## LUNAR_LANDER ##
##################
from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderDqn
config = ConfigLunarLanderDqn()
config.WANDB_REPORt_URL = 'https://wandb.ai/link-koreatech/LunarLander-v2_DQN_anonymous/reports/Lunar-Lander-DQN--VmlldzoxNzI3OTU5?accessToken=4kbmznjvysvlh5zbf30vy8jg403ojrjxjefiewfpcn5iu3cwo5mirclsj8f6dx1u'

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderMuzero
config = ConfigLunarLanderMuzero()

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderA2c
config = ConfigLunarLanderA2c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderA3c
config = ConfigLunarLanderA3c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderPpo
config = ConfigLunarLanderPpo()

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander import ConfigLunarLanderPpoTrajectory
config = ConfigLunarLanderPpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLander-v2_PPO_TRAJECTORY_anonymous/reports/LunarLander-v2-PPO_TRAJECTORY--VmlldzoxNTAwOTg0?accessToken=y2tgu6i38zzehklrq6rikdqhvza2fsexmfg22ntl2drl0q9bsvd11t1b0e09r0ry"

#############################
## LUNAR_LANDER_CONTINUOUS ##
#############################
from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousA2c
config = ConfigLunarLanderContinuousA2c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousA3c
config = ConfigLunarLanderContinuousA3c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousPpo
config = ConfigLunarLanderContinuousPpo()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_PPO_yhhan/reports/LunarLanderContinuous-v2_PPO--VmlldzoxOTkyMDUy?accessToken=von6k6x7zlz1fns9gb7ld7lqpsvtemc3z6yx70t43iuxarppga7v0f0ard4uf18t"

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousPpoTrajectory
config = ConfigLunarLanderContinuousPpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_PPO_TRAJECTORY_yhhan/reports/LunarLanderContinuous-v2_PPO_TRAJECTORY--VmlldzoxNTA3MjAx"

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousDdpg
config = ConfigLunarLanderContinuousDdpg()

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousTd3
config = ConfigLunarLanderContinuousTd3()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_TD3_yhhan/reports/LunarLanderContinuous-v2_TD3--VmlldzoxNTA3MjA4"

from a_configuration.b_single_config.open_ai_gym.box2d.config_lunar_lander_continuous import ConfigLunarLanderContinuousSac
config = ConfigLunarLanderContinuousSac()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/LunarLanderContinuous-v2_SAC_yhhan/reports/LunarLanderContinuous-v2_SAC--VmlldzoxNTA3MjA2"

###########################
## Normal Bipedal Walker ##
###########################
from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerA2c
config = ConfigNormalBipedalWalkerA2c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerA3c
config = ConfigNormalBipedalWalkerA3c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerPpo
config = ConfigNormalBipedalWalkerPpo()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerPpoTrajectory
config = ConfigNormalBipedalWalkerPpoTrajectory()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerDdpg
config = ConfigNormalBipedalWalkerDdpg()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerTd3
config = ConfigNormalBipedalWalkerTd3()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigNormalBipedalWalkerSac
config = ConfigNormalBipedalWalkerSac()

#############################
## Hardcore Bipedal Walker ##
#############################
from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerA2c
config = ConfigHardcoreBipedalWalkerA2c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerA3c
config = ConfigHardcoreBipedalWalkerA3c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerPpo
config = ConfigHardcoreBipedalWalkerPpo()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerPpoTrajectory
config = ConfigHardcoreBipedalWalkerPpoTrajectory()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerDdpg
config = ConfigHardcoreBipedalWalkerDdpg()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerTd3
config = ConfigHardcoreBipedalWalkerTd3()

from a_configuration.b_single_config.open_ai_gym.box2d.config_bipedal_walker import ConfigHardcoreBipedalWalkerSac
config = ConfigHardcoreBipedalWalkerSac()


################
## Car Racing ##
################
from a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingA2c
config = ConfigCarRacingA2c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingA3c
config = ConfigCarRacingA3c()

from a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingPpo
config = ConfigCarRacingPpo()

from a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingPpoTrajectory
config = ConfigCarRacingPpoTrajectory()

from a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingDdpg
config = ConfigCarRacingDdpg()

from a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingTd3
config = ConfigCarRacingTd3()

from a_configuration.b_single_config.open_ai_gym.box2d.config_car_racing import ConfigCarRacingSac
config = ConfigCarRacingSac()


######################
##    ANT_BULLET    ##
######################
from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletA2c
config = ConfigAntBulletA2c()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletPpo
config = ConfigAntBulletPpo()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletPpoTrajectory
config = ConfigAntBulletPpoTrajectory()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletDdpg
config = ConfigAntBulletDdpg()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletTd3
config = ConfigAntBulletTd3()

from a_configuration.b_single_config.pybullet.config_ant_bullet import ConfigAntBulletSac
config = ConfigAntBulletSac()

#########################
##    HOPPER_BULLET    ##
#########################
from a_configuration.b_single_config.pybullet.config_hopper_bullet import ConfigHopperBulletSac
config = ConfigHopperBulletSac()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/HopperBulletEnv-v0_SAC_yhhan/reports/HopperBullet-SAC--VmlldzoxNTI4NTA3?accessToken=4irm57nvvd1ffqf7ggrjt7i47nfckc5d0e0libe5ifsgovz3tr3ewzvt9s2tgr0n"

from a_configuration.b_single_config.pybullet.config_hopper_bullet import ConfigHopperBulletPpo
config = ConfigHopperBulletPpo()

from a_configuration.b_single_config.pybullet.config_hopper_bullet import ConfigHopperBulletPpoTrajectory
config = ConfigHopperBulletPpoTrajectory()


#########################################
##    InvertedDoublePendulumMujoco    ##
#########################################
from a_configuration.b_single_config.open_ai_gym.mujoco.config_inverted_double_pendulum_mujoco import ConfigInvertedDoublePendulumMujocoPpo
config = ConfigInvertedDoublePendulumMujocoPpo()

#########################################
##    InvertedDoublePendulum_BULLET    ##
#########################################
from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletA2c
config = ConfigInvertedDoublePendulumBulletA2c()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletSac
config = ConfigInvertedDoublePendulumBulletSac()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletDdpg
config = ConfigInvertedDoublePendulumBulletDdpg()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletTd3
config = ConfigInvertedDoublePendulumBulletTd3()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletPpo
config = ConfigInvertedDoublePendulumBulletPpo()

from a_configuration.b_single_config.pybullet.config_inverted_double_pendulum_bullet import ConfigInvertedDoublePendulumBulletPpoTrajectory
config = ConfigInvertedDoublePendulumBulletPpoTrajectory()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/InvertedDoublePendulumBulletEnv-v0_PPO_TRAJECTORY_anonymous/reports/Inverted-Double-Pendulum-PPO_Trajectory--VmlldzoxNTAwNzg4?accessToken=j639pzfajkaiybgddbxkp552v0qjcv6zu0ogytbr6ec85qxg7j2gefdh56gvvdx7"

##########
## PONG ##
##########
from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDqn
config = ConfigPongDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDoubleDqn
config = ConfigPongDoubleDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDuelingDqn
config = ConfigPongDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongDoubleDuelingDqn
config = ConfigPongDoubleDuelingDqn()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/Pong-v5_DOUBLE_DUELING_DQN_yhhan/reports/Pong-v5_DOUBLE_DUELING_DQN--VmlldzoxODYxMjU4?accessToken=gsvpknqu1gqi6cb0177jgdzk61oqhpua4kcj680l89fdvwer1cg3wms7rb62syj8"

from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongA2c
config = ConfigPongA2c()

from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongA3c
config = ConfigPongA3c()

from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongPpo
config = ConfigPongPpo()

from a_configuration.b_single_config.open_ai_gym.atari.config_pong import ConfigPongPpoTrajectory
config = ConfigPongPpoTrajectory()

##########
## Breakout ##
##########
from a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutDqn
config = ConfigBreakoutDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutDoubleDqn
config = ConfigBreakoutDoubleDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutDuelingDqn
config = ConfigBreakoutDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutDoubleDuelingDqn
config = ConfigBreakoutDoubleDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutA2c
config = ConfigBreakoutA2c()

from a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutA3c
config = ConfigBreakoutA3c()

from a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutPpo
config = ConfigBreakoutPpo()

from a_configuration.b_single_config.open_ai_gym.atari.config_breakout import ConfigBreakoutPpoTrajectory
config = ConfigBreakoutPpoTrajectory()

##########
## VideoPinball ##
##########
from a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballDqn
config = ConfigVideoPinballDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballDoubleDqn
config = ConfigVideoPinballDoubleDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballDuelingDqn
config = ConfigVideoPinballDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballDoubleDuelingDqn
config = ConfigVideoPinballDoubleDuelingDqn()

from a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballA2c
config = ConfigVideoPinballA2c()

from a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballA3c
config = ConfigVideoPinballA3c()

from a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballPpo
config = ConfigVideoPinballPpo()

from a_configuration.b_single_config.open_ai_gym.atari.config_video_pinball import ConfigVideoPinballPpoTrajectory
config = ConfigVideoPinballPpoTrajectory()

##################
### ANT_MUJOCO ###
##################
from a_configuration.b_single_config.open_ai_gym.mujoco.config_ant_mujoco import ConfigAntMujocoSac
config = ConfigAntMujocoSac()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/Ant-v2_SAC_yhhan/reports/Ant-v2_SAC--VmlldzoxNTM0NDMz"

from a_configuration.b_single_config.open_ai_gym.mujoco.config_ant_mujoco import ConfigAntMujocoPpo
config = ConfigAntMujocoPpo()

from a_configuration.b_single_config.open_ai_gym.mujoco.config_ant_mujoco import ConfigAntMujocoPpoTrajectory
config = ConfigAntMujocoPpoTrajectory()

#####################
### HOPPER_MUJOCO ###
#####################
from a_configuration.b_single_config.open_ai_gym.mujoco.config_hopper_mujoco import ConfigHopperMujocoSac
config = ConfigHopperMujocoSac()

from a_configuration.b_single_config.open_ai_gym.mujoco.config_hopper_mujoco import ConfigHopperMujocoPpo
config = ConfigHopperMujocoPpo()

from a_configuration.b_single_config.open_ai_gym.mujoco.config_hopper_mujoco import ConfigHopperMujocoPpoTrajectory
config = ConfigHopperMujocoPpoTrajectory()

#######################
### WALKER2d_MUJOCO ###
#######################
from a_configuration.b_single_config.open_ai_gym.mujoco.config_walker2d_mujoco import ConfigWalker2dMujocoSac
config = ConfigWalker2dMujocoSac()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/Walker2d-v2_SAC_yhhan/reports/3DBall-SAC--VmlldzoxNTI4NDk4?accessToken=ikha9ymzoqh7o3zefs6cn0ao5hmz7qzc8wnx0shd7i63wesx8585ja901i1bie1z"

from a_configuration.b_single_config.open_ai_gym.mujoco.config_walker2d_mujoco import ConfigWalker2dMujocoPpo
config = ConfigWalker2dMujocoPpo()

from a_configuration.b_single_config.open_ai_gym.mujoco.config_walker2d_mujoco import ConfigWalker2dMujocoPpoTrajectory
config = ConfigWalker2dMujocoPpoTrajectory()

##########################
### HALFCHEETAH_MUJOCO ###
##########################
from a_configuration.b_single_config.open_ai_gym.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoSac
config = ConfigHalfCheetahMujocoSac()

from a_configuration.b_single_config.open_ai_gym.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoPpo
config = ConfigHalfCheetahMujocoPpo()

from a_configuration.b_single_config.open_ai_gym.mujoco.config_halfcheetah_mujoco import ConfigHalfCheetahMujocoPpoTrajectory
config = ConfigHalfCheetahMujocoPpoTrajectory()

##########################
### Unity3DBall ###
##########################
from a_configuration.b_single_config.unity.config_3d_ball import Config3DBallDdqg
config = Config3DBallDdqg()
config.WANDB_REPORT_URL = "https://wandb.ai/link-koreatech/Unity3DBall_DDPG_yhhan/reports/3DBall-Ddqg--VmlldzoxNTM0NTUy?accessToken=gxnxojzda8hu41wb52vv873kql0qkir16nyqfcrrzq6eeaozen9xfe2pmg4f0dt5"


##########################
### UnityWalker ###
##########################
from a_configuration.b_single_config.unity.config_walker import ConfigWalkerSac
config = ConfigWalkerSac()


##########################
### UnityDrone ###
##########################
from a_configuration.b_single_config.unity.config_drone import ConfigDroneDdpg
config = ConfigDroneDdpg()

from a_configuration.b_single_config.unity.config_drone import ConfigDroneSac
config = ConfigDroneSac()

###########################
#### Knapsack - Random ####
###########################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0RandomTestDqn
config = ConfigKnapsack0RandomTestDqn()


##################################
#### Knapsack - Random Linear ####
##################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0RandomTestLinearDqn
config = ConfigKnapsack0RandomTestLinearDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0RandomTestLinearA2c
config = ConfigKnapsack0RandomTestLinearA2c()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0RandomTestLinearPpo
config = ConfigKnapsack0RandomTestLinearPpo()


#################################
######## Knapsack - Load ########
#################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0LoadTestDqn
config = ConfigKnapsack0LoadTestDqn()


#################################
#### Knapsack - Load Linear #####
#################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0LoadTestLinearDqn
config = ConfigKnapsack0LoadTestLinearDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0LoadTestLinearA2c
config = ConfigKnapsack0LoadTestLinearA2c()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0LoadTestLinearPpo
config = ConfigKnapsack0LoadTestLinearPpo()


#################################
####### Knapsack - static #######
#################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestDqn
config = ConfigKnapsack0StaticTestDqn()



########################################
####### Knapsack - Static Linear #######
########################################

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearDqn
config = ConfigKnapsack0StaticTestLinearDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearDoubleDqn
config = ConfigKnapsack0StaticTestLinearDoubleDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearDoubleDuelingDqn
config = ConfigKnapsack0StaticTestLinearDoubleDuelingDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearA2c
config = ConfigKnapsack0StaticTestLinearA2c()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack0StaticTestLinearPpo
config = ConfigKnapsack0StaticTestLinearPpo()


##################################
##Action_Space = NUMBER_OF_ITEMS##
####### Knapsack - Random ########
##################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1RandomTestDqn
config = ConfigKnapsack1RandomTestDqn()


#################################
#### Knapsack - Random Linear ####
#################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1RandomTestLinearDqn
config = ConfigKnapsack1RandomTestLinearDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1RandomTestLinearA2c
config = ConfigKnapsack1RandomTestLinearA2c()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1RandomTestLinearPpo
config = ConfigKnapsack1RandomTestLinearPpo()


#################################
######## Knapsack - Load ########
#################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1LoadTestDqn
config = ConfigKnapsack1LoadTestDqn()


#################################
#### Knapsack - Load Linear #####
#################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1LoadTestLinearDqn
config = ConfigKnapsack1LoadTestLinearDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1LoadTestLinearA2c
config = ConfigKnapsack1LoadTestLinearA2c()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1LoadTestLinearPpo
config = ConfigKnapsack1LoadTestLinearPpo()


#################################
####### Knapsack - static #######
#################################
from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1StaticTestDqn
config = ConfigKnapsack1StaticTestDqn()

########################################
####### Knapsack - Static Linear #######
########################################

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1StaticTestLinearDqn
config = ConfigKnapsack1StaticTestLinearDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1StaticTestLinearDoubleDqn
config = ConfigKnapsack1StaticTestLinearDoubleDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1StaticTestLinearDoubleDuelingDqn
config = ConfigKnapsack1StaticTestLinearDoubleDuelingDqn()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1StaticTestLinearA2c
config = ConfigKnapsack1StaticTestLinearA2c()

from a_configuration.b_single_config.combinatorial_optimization.config_knapsack import ConfigKnapsack1StaticTestLinearPpo
config = ConfigKnapsack1StaticTestLinearPpo()


###########################################
####### HerKnapsack - Static Linear #######
###########################################

from a_configuration.b_single_config.combinatorial_optimization.config_her_knapsack import ConfigHerKnapsack0StaticTestLinearDoubleDqn
config = ConfigHerKnapsack0StaticTestLinearDoubleDqn()


###########################
### BasicTaskAllocation ###
###########################
from a_configuration.b_single_config.combinatorial_optimization.config_basic_task_allocation import ConfigBasicTaskAllocation0Dqn
config = ConfigBasicTaskAllocation0Dqn()

from a_configuration.b_single_config.combinatorial_optimization.config_basic_task_allocation import ConfigBasicTaskAllocation1Dqn
config = ConfigBasicTaskAllocation1Dqn()


###############################
####### Task Allocation #######
###############################
from a_configuration.b_single_config.task_allocation.config_task_allocation import ConfigTaskAllocationDqn
config = ConfigTaskAllocationDqn()

config.USE_WANDB = False

if __name__ == "__main__":
    from g_utils.commons import print_basic_info, get_env_info, set_config

    set_config(config)
    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)