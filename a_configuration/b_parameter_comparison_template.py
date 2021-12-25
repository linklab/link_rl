from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_cartpole import ParameterComparisonCartPoleDqn
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_pong import ParameterComparisonPongDqn
from a_configuration.d_parameters_comparison.pybullet.parameter_comparison_cartpole_bullet import \
    ParameterComparisonCartPoleBulletA2c

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
for parameter_comparison in parameter_comparison_list:
    parameter_comparison.USE_WANDB = False
    parameter_comparison.WANDB_ENTITY = "link-koreatech"

