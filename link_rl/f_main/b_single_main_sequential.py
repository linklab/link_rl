from b_single_main_common import *
from link_rl.h_utils.commons import model_load, get_specific_env_name, print_model_summary
from link_rl.h_utils.types import AgentType, ActorCriticAgentTypes


def main():
    set_config(config)

    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)

    agent = get_agent(
        observation_space=observation_space, action_space=action_space, config=config
    )

    #print_model_summary(agent=agent, observation_space=observation_space, action_space=action_space, config=config)

    env_name = get_specific_env_name(config=config)

    model_load(agent=agent, env_name=env_name, agent_type_name=config.AGENT_TYPE.name, config=config)

    learner = Learner(agent=agent, queue=None, config=config)

    print("########## LEARNING STARTED !!! ##########")
    learner.train_loop()

    print_basic_info(observation_space, action_space, config)


if __name__ == "__main__":
    assert config.AGENT_TYPE not in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]
    assert config.N_ACTORS == 1, "Current config.N_ACTORS: {0}".format(config.N_ACTORS)
    main()
