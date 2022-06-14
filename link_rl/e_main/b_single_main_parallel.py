import time
import torch.multiprocessing as mp
from link_rl.d_agents.on_policy.a3c.agent_a3c import WorkerAgentA3c
from link_rl.d_agents.on_policy.asynchronous_ppo.agent_asynchronous_ppo import WorkerAsynchronousPpo

from link_rl.e_main.supports.actor import Actor, LearningActor
from link_rl.g_utils.types import OffPolicyAgentTypes, AgentType


def main():
    set_config(config)

    observation_space, action_space = get_env_info(config)
    print_basic_info(observation_space, action_space, config)

    input("Press Enter (two or more times) to continue...")

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()

    if config.AGENT_TYPE in [AgentType.A3C, AgentType.ASYNCHRONOUS_PPO]:
        master_agent = get_agent(
            observation_space=observation_space, action_space=action_space, config=config
        )

        shared_model_access_lock = mp.Lock()

        learner = Learner(
            agent=master_agent, queue=queue, shared_model_access_lock=shared_model_access_lock, config=config
        )

        if config.AGENT_TYPE == AgentType.A3C:
            worker_agent_class = WorkerAgentA3c
        elif config.AGENT_TYPE == AgentType.ASYNCHRONOUS_PPO:
            worker_agent_class = WorkerAsynchronousPpo
        else:
            raise ValueError()

        worker_agents = [
            worker_agent_class(
                master_agent=master_agent, observation_space=observation_space, action_space=action_space,
                shared_model_access_lock=shared_model_access_lock, config=config
            ) for _ in range(config.N_ACTORS)
        ]

        actors = [
            LearningActor(
                env_name=config.ENV_NAME, actor_id=actor_id, agent=worker_agents[actor_id], queue=queue, config=config
            ) for actor_id in range(config.N_ACTORS)
        ]
    else:
        agent = get_agent(
            observation_space=observation_space, action_space=action_space, config=config
        )

        learner = Learner(agent=agent, queue=queue, config=config)

        actors = [
            Actor(
                env_name=config.ENV_NAME, actor_id=actor_id, agent=agent, queue=queue, config=config
            ) for actor_id in range(config.N_ACTORS)
        ]

    for actor in actors:
        actor.start()

    # Busy Wait: 모든 액터들이 VecEnv를 생성 완료할 때까지 대기
    for actor in actors:
        while not actor.is_vectorized_env_created.value:
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
