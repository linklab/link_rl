import time
import torch.multiprocessing as mp
from link_rl.d_agents.on_policy.a3c.agent_a3c import WorkingAgentA3c
from link_rl.d_agents.on_policy.asynchronous_ppo.agent_asynchronous_ppo import WorkingAsynchronousPpo

from link_rl.e_main.supports.actor import Actor
from link_rl.g_utils.commons import get_specific_env_name, model_load, print_model_summary
from link_rl.g_utils.types import AgentType, ActorCriticAgentTypes

from b_single_main_common import *


def main():
    set_config(config)
    config.TRAIN_INTERVAL_GLOBAL_TIME_STEPS = 1

    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()

    if config.AGENT_TYPE in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]:
        master_agent = get_agent(
            observation_space=observation_space, action_space=action_space, config=config
        )

        print_model_summary(
            agent=master_agent, observation_space=observation_space, action_space=action_space, config=config
        )

        shared_model_access_lock = mp.Lock()

        learner = Learner(
            agent=master_agent, queue=queue, shared_model_access_lock=shared_model_access_lock, config=config
        )

        if config.AGENT_TYPE == AgentType.A3C:
            working_agent_class = WorkingAgentA3c
        elif config.AGENT_TYPE == AgentType.ASYNCHRONOUS_PPO:
            working_agent_class = WorkingAsynchronousPpo
        else:
            raise ValueError()

        working_agents = [
            working_agent_class(
                master_agent=master_agent, observation_space=observation_space, action_space=action_space,
                shared_model_access_lock=shared_model_access_lock, config=config, need_train=True
            ) for _ in range(config.N_ACTORS)
        ]

        env_name = get_specific_env_name(config=config)

        model_load(agent=working_agents, env_name=env_name, agent_type_name=config.AGENT_TYPE.name, config=config)

        actors = [
            Actor(
                actor_id=actor_id, agent=working_agents[actor_id], queue=queue, config=config, working_actor=True
            ) for actor_id in range(config.N_ACTORS)
        ]
    else:
        agent = get_agent(
            observation_space=observation_space, action_space=action_space, config=config
        )

        print_model_summary(agent=agent, observation_space=observation_space, action_space=action_space, config=config)

        env_name = get_specific_env_name(config=config)

        model_load(agent=agent, env_name=env_name, agent_type_name=config.AGENT_TYPE.name, config=config)

        learner = Learner(agent=agent, queue=queue, config=config)

        actors = [
            Actor(
                actor_id=actor_id, agent=agent, queue=queue, config=config
            ) for actor_id in range(config.N_ACTORS)
        ]

    for actor in actors:
        actor.start()

    # Busy Wait: 모든 액터들이 VecEnv를 생성 완료할 때까지 대기
    for actor in actors:
        while not actor.is_env_created.value:
            time.sleep(0.1)

    print("########## LEARNING STARTED !!! ##########")

    learner.start()

    while True:
        # Busy Wait: learner에서 학습 완료될 때까지 대기
        if learner.is_terminated.value:
            # learner가 학습 완료하면 각 actor들의 rollout 종료
            for actor in actors:
                actor.is_terminated.value = True
            break
        time.sleep(0.5)

    # Busy Wait: 모든 actor가 조인할 때까지 대기
    while any([actor.is_alive() for actor in actors]):
        for actor in actors:
            actor.join(timeout=1)

    # Busy Wait: learner가 조인할 때까지 대기
    while learner.is_alive():
        learner.join(timeout=1)

    print_basic_info(observation_space, action_space, config)


if __name__ == "__main__":
    assert config.AGENT_TYPE in OffPolicyAgentTypes or config.AGENT_TYPE in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]
    main()
