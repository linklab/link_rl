import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.supports.main_preamble import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parameter = Parameter()


def main():
    print_basic_info(device, parameter)

    obs_shape, n_actions = get_env_info(parameter)

    agent = get_agent(obs_shape, n_actions, device, parameter)

    learner = Learner(
        agent=agent, queue=None, device=device, parameter=parameter
    )

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop(parallel=False)

    print_basic_info(device, parameter)


if __name__ == "__main__":
    assert parameter.N_ACTORS == 1
    main()
