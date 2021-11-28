import os
import sys

from g_utils.commons import get_wandb_obj

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.supports.main_preamble import *


def main():
    print_basic_info(device, params)

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()

    learner = Learner(
        test_env=test_env,
        agent=agent,
        queue=queue,
        device=device,
        params=params,
    )

    actors = [
        Actor(
            env_name=params.ENV_NAME,
            actor_id=actor_id,
            agent=agent,
            queue=queue,
            params=params
        ) for actor_id in range(params.N_ACTORS)
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

    print_basic_info(device, params)


if __name__ == "__main__":
    assert params.AGENT_TYPE in OffPolicyAgentTypes
    main()
