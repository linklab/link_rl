import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.supports.main_preamble import *


def main():
    print_basic_info(device, params)

    test_env, obs_shape, n_actions = get_test_env(params)

    agent = get_agent(obs_shape, n_actions, device, params)

    learner = Learner(
        test_env=test_env, agent=agent,
        queue=None, device=device, params=params
    )

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop(sync=True)

    print_basic_info(device, params)


if __name__ == "__main__":
    # assert params.AGENT_TYPE in OnPolicyAgentTypes
    assert params.N_ACTORS == 1
    main()
