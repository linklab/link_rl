import os
import sys

from e_main.supports.learnerComparison import LearnerComparison

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.supports.main_preamble import *
from a_configuration.parameter_comparison import ParameterComparison
from g_utils.commons import get_wandb_obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parameter_comparison = ParameterComparison()

n_agents = len(parameter_comparison.parameters)


def main():
    print_basic_info(device, parameter_comparison)

    obs_shape, n_actions = get_env_info(parameter_comparison)

    agents = []
    for idx in range(n_agents):
        agent = get_agent(
            obs_shape, n_actions, device, parameter_comparison.parameters[idx]
        )
        agents.append(agent)

    learner_comparison = LearnerComparison(
        n_agents=n_agents, agents=agents, device=device,
        parameter_comparison=parameter_comparison
    )

    print("########## LEARNING STARTED !!! ##########")
    learner_comparison.train_loop(sync=True)

    print_basic_info(device, parameter)


if __name__ == "__main__":
    # assert parameter.AGENT_TYPE in OnPolicyAgentTypes
    assert parameter_comparison.N_ACTORS == 1
    main()
