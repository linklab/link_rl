import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from a_configuration.parameter_comparison import ParameterComparison
from e_main.supports.main_preamble import *
from e_main.supports.learner_comparison import LearnerComparison
from g_utils.commons import print_comparison_basic_info, get_wandb_obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parameter_c = ParameterComparison()

n_agents = len(parameter_c.AGENT_PARAMETERS)


def main():
    if parameter_c.USE_WANDB:
        wandb_obj = get_wandb_obj(parameter_c, comparison=True)
    else:
        wandb_obj = None

    print_comparison_basic_info(device, parameter_c)

    obs_shape, n_actions = get_env_info(parameter_c)

    agents = []
    for agent_idx, _ in enumerate(parameter_c.AGENT_PARAMETERS):
        agent = get_agent(
            obs_shape, n_actions, device, parameter_c.AGENT_PARAMETERS[agent_idx], parameter_c.MAX_TRAINING_STEPS
        )
        agents.append(agent)

    print("########## LEARNING STARTED !!! ##########")
    for run in range(0, parameter_c.N_RUNS):
        print(">" * 30 + " RUN: {0} ".format(run + 1) + "<" * 30)
        learner_comparison = LearnerComparison(
            run=run, agents=agents, device=device, wandb_obj=wandb_obj, parameter_c=parameter_c
        )
        learner_comparison.train_loop()

    if parameter_c.USE_WANDB:
        wandb_obj.join()

    print_comparison_basic_info(device, parameter_c)


if __name__ == "__main__":
    # assert parameter.AGENT_TYPE in OnPolicyAgentTypes
    assert parameter_c.N_ACTORS == 1 and parameter_c.N_VECTORIZED_ENVS == 1
    main()
