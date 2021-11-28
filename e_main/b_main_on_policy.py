import os
import sys
from gym.vector import AsyncVectorEnv

from b_environments.make_envs import make_gym_env

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.supports.main_preamble import *


def main():
    print_basic_info(device, params)

    train_env = AsyncVectorEnv(
        env_fns=[
            make_gym_env(params.ENV_NAME) for _ in range(params.N_VECTORIZED_ENVS)
        ]
    )

    agent = get_agent(n_features, n_actions, device, params)

    learner = Learner(
        test_env=test_env,
        agent=agent,
        queue=None,
        device=device,
        params=params,
        train_env=train_env
    )

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop()

    print_basic_info(device, params)


if __name__ == "__main__":
    assert params.AGENT_TYPE in OnPolicyAgentTypes
    assert params.N_ACTORS == 1
    main()