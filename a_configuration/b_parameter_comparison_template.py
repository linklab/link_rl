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


parameter_c = ParameterComparisonCartPoleDqn()

################################################################################

parameter_c = ParameterComparisonPongDqn()

################################################################################

parameter_c = ParameterComparisonCartPoleBulletA2c()


#######################################################################

parameter_c = ParameterComparisonCartPoleDqnTypes()


######################################################################

parameter_c = ParameterComparisonCartPoleBulletDqnTypes()


#######################################################################################################################
parameter_c = ParameterComparisonCartPoleContinuousBulletDdpg()

######################################################################

parameter_c = ParameterComparisonPongDqnTypes()


######################################################################

parameter_c = ParameterComparisonCartPoleDqn()


######################################################################

parameter_comparison_lunar_lander_recurrent = ParameterComparisonLunarLanderDqnRecurrent()
parameter_comparison_lunar_lander_recurrent.AGENT_LABELS = [
    "DQN Small Linear",
    "DQN Small Recurrent",
]

parameter_comparison_lunar_lander_recurrent.MAX_TRAINING_STEPS = 100_000
parameter_comparison_lunar_lander_recurrent.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_lunar_lander_recurrent)

######################################################################

parameter_comparison_mujoco_hopper_alpha = ParameterComparisonHopperMujocoSac()
parameter_comparison_mujoco_hopper_alpha.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_mujoco_hopper_alpha.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
parameter_comparison_mujoco_hopper_alpha.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_mujoco_hopper_alpha.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
parameter_comparison_mujoco_hopper_alpha.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_mujoco_hopper_alpha.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
parameter_comparison_mujoco_hopper_alpha.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_mujoco_hopper_alpha.AGENT_PARAMETERS[3].MIN_ALPHA = 0.0
parameter_comparison_mujoco_hopper_alpha.AGENT_LABELS = [
    "alpha = 0.2",
    "alpha = 0.5",
    "alpha tuning (No Alpha Limit)",
    "alpha tuning (Min Alpha = 0.2)",
]
parameter_comparison_mujoco_hopper_alpha.MAX_TRAINING_STEPS = 300000
parameter_comparison_mujoco_hopper_alpha.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_mujoco_hopper_alpha)

######################################################################

parameter_comparison_mujoco_walker2d_alpha = ParameterComparisonWalker2dMujocoSac()
parameter_comparison_mujoco_walker2d_alpha.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_mujoco_walker2d_alpha.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
parameter_comparison_mujoco_walker2d_alpha.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_mujoco_walker2d_alpha.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
parameter_comparison_mujoco_walker2d_alpha.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_mujoco_walker2d_alpha.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
parameter_comparison_mujoco_walker2d_alpha.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_mujoco_walker2d_alpha.AGENT_PARAMETERS[3].MIN_ALPHA = 0.2
parameter_comparison_mujoco_walker2d_alpha.AGENT_LABELS = [
    "alpha = 0.2",
    "alpha = 0.5",
    "alpha tuning (No Alpha Limit)",
    "alpha tuning (Min Alpha = 0.2)",
]
parameter_comparison_mujoco_walker2d_alpha.MAX_TRAINING_STEPS = 700000
parameter_comparison_mujoco_walker2d_alpha.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_mujoco_walker2d_alpha)

######################################################################

parameter_comparison_halfcheetah_mujoco_alpha = ParameterComparisonHalfCheetahMujocoSac()
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_PARAMETERS[3].MIN_ALPHA = 0.2
parameter_comparison_halfcheetah_mujoco_alpha.AGENT_LABELS = [
    "alpha = 0.2",
    "alpha = 0.5",
    "alpha tuning (No Alpha Limit)",
    "alpha tuning (Min Alpha = 0.2)",
]
parameter_comparison_halfcheetah_mujoco_alpha.MAX_TRAINING_STEPS = 500000
parameter_comparison_halfcheetah_mujoco_alpha.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_halfcheetah_mujoco_alpha)

######################################################################

parameter_comparison_ant_mujoco_alpha = ParameterComparisonAntMujocoSac()
parameter_comparison_ant_mujoco_alpha.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_ant_mujoco_alpha.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
parameter_comparison_ant_mujoco_alpha.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_ant_mujoco_alpha.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
parameter_comparison_ant_mujoco_alpha.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_ant_mujoco_alpha.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
parameter_comparison_ant_mujoco_alpha.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_ant_mujoco_alpha.AGENT_PARAMETERS[3].MIN_ALPHA = 0.2
parameter_comparison_ant_mujoco_alpha.AGENT_LABELS = [
    "alpha = 0.2",
    "alpha = 0.5",
    "alpha tuning (No Alpha Limit)",
    "alpha tuning (Min Alpha = 0.2)",
]
parameter_comparison_ant_mujoco_alpha.MAX_TRAINING_STEPS = 500000
parameter_comparison_ant_mujoco_alpha.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_ant_mujoco_alpha)

######################################################################

parameter_comparison_ant_bullet_alpha = ParameterComparisonAntBulletSac()
parameter_comparison_ant_bullet_alpha.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_ant_bullet_alpha.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
parameter_comparison_ant_bullet_alpha.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_ant_bullet_alpha.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
parameter_comparison_ant_bullet_alpha.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_ant_bullet_alpha.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
parameter_comparison_ant_bullet_alpha.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_ant_bullet_alpha.AGENT_PARAMETERS[3].MIN_ALPHA = 0.2
parameter_comparison_ant_bullet_alpha.AGENT_LABELS = [
    "alpha = 0.2",
    "alpha = 0.5",
    "alpha tuning (No Alpha Limit)",
    "alpha tuning (Min Alpha = 0.2)",
]
parameter_comparison_ant_bullet_alpha.MAX_TRAINING_STEPS = 500000
parameter_comparison_ant_bullet_alpha.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_ant_bullet_alpha)

######################################################################

parameter_comparison_double_inverted_pendulum_bullet_alpha = ParameterComparisonDoubleInvertedPendulumBulletSac()
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_PARAMETERS[3].MIN_ALPHA = 0.2
parameter_comparison_double_inverted_pendulum_bullet_alpha.AGENT_LABELS = [
    "alpha = 0.2",
    "alpha = 0.5",
    "alpha tuning (No Alpha Limit)",
    "alpha tuning (Min Alpha = 0.2)",
]
parameter_comparison_double_inverted_pendulum_bullet_alpha.MAX_TRAINING_STEPS = 100000
parameter_comparison_double_inverted_pendulum_bullet_alpha.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_double_inverted_pendulum_bullet_alpha)

#######################################################################
for parameter_comparison in parameter_comparison_list:
    parameter_comparison.USE_WANDB = False
    parameter_comparison.WANDB_ENTITY = "link-koreatech"

