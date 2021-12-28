import os
import sys
import warnings

from g_utils.stats import ComparisonStat

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.parameter_comparison import parameter_c
from e_main.supports.main_preamble import *
from e_main.supports.learner_comparison import LearnerComparison
from g_utils.commons import print_comparison_basic_info, get_wandb_obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_agents = len(parameter_c.AGENT_PARAMETERS)

for agent_parameter in parameter_c.AGENT_PARAMETERS:
    del agent_parameter.MAX_TRAINING_STEPS
    del agent_parameter.N_ACTORS
    del agent_parameter.N_EPISODES_FOR_MEAN_CALCULATION
    del agent_parameter.N_TEST_EPISODES
    del agent_parameter.N_VECTORIZED_ENVS
    del agent_parameter.PROJECT_HOME
    del agent_parameter.TEST_INTERVAL_TRAINING_STEPS
    del agent_parameter.TRAIN_INTERVAL_GLOBAL_TIME_STEPS
    del agent_parameter.USE_WANDB
    del agent_parameter.WANDB_ENTITY
    del agent_parameter.MODEL_SAVE_DIR
    del agent_parameter.CONSOLE_LOG_INTERVAL_TRAINING_STEPS


def main():
    observation_space, action_space = get_env_info(parameter_c)
    print_comparison_basic_info(observation_space, action_space, device, parameter_c)

    input("Press Enter to continue...")

    if parameter_c.USE_WANDB:
        wandb_obj = get_wandb_obj(parameter_c, comparison=True)
    else:
        wandb_obj = None

    comparison_stat = ComparisonStat(parameter_c=parameter_c)

    print("\n########## LEARNING STARTED !!! ##########")
    for run in range(0, parameter_c.N_RUNS):
        print("\n" + ">" * 30 + " RUN: {0} ".format(run + 1) + "<" * 30)
        agents = []
        for agent_idx, _ in enumerate(parameter_c.AGENT_PARAMETERS):
            agent = get_agent(
                observation_space=observation_space, action_space=action_space, device=device,
                parameter=parameter_c.AGENT_PARAMETERS[agent_idx]
            )
            agents.append(agent)

        learner_comparison = LearnerComparison(
            run=run, agents=agents, device=device, wandb_obj=wandb_obj,
            parameter_c=parameter_c, comparison_stat=comparison_stat
        )
        learner_comparison.train_comparison_loop()

    if parameter_c.USE_WANDB:
        wandb_obj.join()

    print_comparison_basic_info(observation_space, action_space, device, parameter_c)

    comparison_stat.save_csv()
    comparison_stat.save_fig()


if __name__ == "__main__":
    # assert parameter.AGENT_TYPE in OnPolicyAgentTypes
    assert parameter_c.N_ACTORS == 1 and parameter_c.N_VECTORIZED_ENVS == 1
    main()
