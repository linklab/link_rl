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
    observation_space, action_space = get_env_info(parameter)
    print_basic_info(observation_space, action_space, device, parameter)

    input("Press Enter to continue...")

    agent = get_agent(
        observation_space, action_space, device, parameter, parameter.MAX_TRAINING_STEPS
    )

    learner = Learner(
        agent=agent, queue=None, device=device, parameter=parameter
    )

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop(parallel=False)

    print_basic_info(observation_space, action_space, device, parameter)


if __name__ == "__main__":
    assert parameter.N_ACTORS == 1
    main()
