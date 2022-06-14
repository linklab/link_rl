from b_single_main_common import *


def main():
    set_config(config)

    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)

    input("Press Enter (two or more times) to continue...")

    agent = get_agent(
        observation_space=observation_space, action_space=action_space, config=config
    )

    learner = Learner(agent=agent, queue=None, config=config)

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop(parallel=False)

    print_basic_info(observation_space, action_space, config)


if __name__ == "__main__":
    assert config.N_ACTORS == 1, "Current config.N_ACTORS: {0}".format(config.N_ACTORS)
    main()
