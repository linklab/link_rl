from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_ant import ParameterComparisonAntMujocoSac
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_halfcheetah import \
    ParameterComparisonHalfCheetahMujocoSac
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_hopper import \
    ParameterComparisonHopperMujocoSac
from a_configuration.d_parameters_comparison.mujoco.parameter_comparison_walker2d import \
    ParameterComparisonWalker2dMujocoSac
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_cartpole import \
    ParameterComparisonCartPoleDqn, ParameterComparisonCartPoleDqnTypes
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_pong import ParameterComparisonPongDqn
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_ant import ParameterComparisonAntBulletSac
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_cartpole_bullet import \
    ParameterComparisonCartPoleBulletA2c
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_double_inverted_pendulum_bullet import \
    ParameterComparisonDoubleInvertedPendulumBulletSac

parameter_comparison_list = []

parameter_comparison_cart_pole_dqn = ParameterComparisonCartPoleDqn()
parameter_comparison_cart_pole_dqn.AGENT_PARAMETERS[0].N_STEP = 1
parameter_comparison_cart_pole_dqn.AGENT_PARAMETERS[1].N_STEP = 2
parameter_comparison_cart_pole_dqn.AGENT_PARAMETERS[2].N_STEP = 4
parameter_comparison_cart_pole_dqn.AGENT_LABELS = [
    "DQN (N_STEP=1)",
    "DQN (N_STEP=2)",
    "DQN (N_STEP=4)",
]
parameter_comparison_cart_pole_dqn.MAX_TRAINING_STEPS = 50_000
parameter_comparison_cart_pole_dqn.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_cart_pole_dqn)

################################################################################

parameter_comparison_pong_dqn = ParameterComparisonPongDqn()
parameter_comparison_pong_dqn.AGENT_PARAMETERS[0].N_STEP = 1
parameter_comparison_pong_dqn.AGENT_PARAMETERS[1].N_STEP = 2
parameter_comparison_pong_dqn.AGENT_PARAMETERS[2].N_STEP = 3
parameter_comparison_pong_dqn.AGENT_LABELS = [
    "DQN (N_STEP=1)",
    "DQN (N_STEP=2)",
    "DQN (N_STEP=3)",
]
parameter_comparison_pong_dqn.MAX_TRAINING_STEPS = 1_000
parameter_comparison_pong_dqn.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_pong_dqn)

################################################################################

parameter_comparison_list = []

parameter_comparison_cart_pole_bullet_a2c = ParameterComparisonCartPoleBulletA2c()
parameter_comparison_cart_pole_bullet_a2c.AGENT_PARAMETERS[0].LEARNING_RATE = 0.001
parameter_comparison_cart_pole_bullet_a2c.AGENT_PARAMETERS[1].LEARNING_RATE = 0.0001
parameter_comparison_cart_pole_bullet_a2c.AGENT_PARAMETERS[2].LEARNING_RATE = 0.00001
parameter_comparison_cart_pole_bullet_a2c.AGENT_LABELS = [
    "DQN (LEARNING_RATE = 0.001)",
    "DQN (LEARNING_RATE = 0.0001)",
    "DQN (LEARNING_RATE = 0.00001)",
]
parameter_comparison_cart_pole_dqn.MAX_TRAINING_STEPS = 50_000
parameter_comparison_cart_pole_dqn.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_cart_pole_bullet_a2c)

#######################################################################

parameter_comparison_cart_pole_dqn_types = ParameterComparisonCartPoleDqnTypes()
parameter_comparison_cart_pole_dqn_types.AGENT_LABELS = [
    "DQN",
    "Double DQN",
    "Dueling DQN",
    "Double Dueling DQN",
]
parameter_comparison_cart_pole_dqn_types.MAX_TRAINING_STEPS = 50_000
parameter_comparison_cart_pole_dqn_types.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_cart_pole_dqn_types)

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

parameter_comparison_doubleinvertedpendulum_bullet_alpha = ParameterComparisonDoubleInvertedPendulumBulletSac()
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_PARAMETERS[0].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_PARAMETERS[0].DEFAULT_ALPHA = 0.2
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_PARAMETERS[1].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = False
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_PARAMETERS[1].DEFAULT_ALPHA = 0.5
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_PARAMETERS[2].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_PARAMETERS[2].MIN_ALPHA = 0.0
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_PARAMETERS[3].AUTOMATIC_ENTROPY_TEMPERATURE_TUNING = True
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_PARAMETERS[3].MIN_ALPHA = 0.2
parameter_comparison_doubleinvertedpendulum_bullet_alpha.AGENT_LABELS = [
    "alpha = 0.2",
    "alpha = 0.5",
    "alpha tuning (No Alpha Limit)",
    "alpha tuning (Min Alpha = 0.2)",
]
parameter_comparison_doubleinvertedpendulum_bullet_alpha.MAX_TRAINING_STEPS = 100000
parameter_comparison_doubleinvertedpendulum_bullet_alpha.N_RUNS = 5
parameter_comparison_list.append(parameter_comparison_doubleinvertedpendulum_bullet_alpha)

#######################################################################
for parameter_comparison in parameter_comparison_list:
    parameter_comparison.USE_WANDB = False
    parameter_comparison.WANDB_ENTITY = "link-koreatech"

