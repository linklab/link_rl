import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from a_configuration.parameter_comparison import ParameterComparison
from e_main.supports.main_preamble import *
from e_main.supports.learner_comparison import LearnerComparison
from g_utils.commons import print_comparison_basic_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parameter_c = ParameterComparison()

n_agents = len(parameter_c.AGENT_PARAMETERS)


def main():
    print_comparison_basic_info(device, parameter_c)

    obs_shape, n_actions = get_env_info(parameter_c)

    agents = []
    for agent_idx, _ in enumerate(parameter_c.AGENT_PARAMETERS):
        agent = get_agent(
            obs_shape, n_actions, device, parameter_c.AGENT_PARAMETERS[agent_idx], parameter_c.MAX_TRAINING_STEPS
        )
        agents.append(agent)

    learner_comparison = LearnerComparison(
        n_agents=n_agents, agents=agents, device=device, parameter_c=parameter_c
    )

    print("########## LEARNING STARTED !!! ##########")
    learner_comparison.train_loop()

    print_basic_info(device, parameter_c)


if __name__ == "__main__":
    # assert parameter.AGENT_TYPE in OnPolicyAgentTypes
    assert parameter_c.N_ACTORS == 1
    main()
