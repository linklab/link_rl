from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_cartpole import ParameterComparisonCartPoleDqn
from a_configuration.d_parameters_comparison.open_ai_gym.parameter_comparison_pong import ParameterComparisonPongDqn

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


#######################################################################
for parameter_comparison in parameter_comparison_list:
    parameter_comparison.USE_WANDB = False
    parameter_comparison.WANDB_ENTITY = "link-koreatech"

