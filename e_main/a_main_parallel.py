import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.parameter import parameter
from e_main.supports.main_preamble import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    observation_space, action_space = get_env_info(parameter)
    print_basic_info(observation_space, action_space, device, parameter)

    input("Press Enter to continue...")

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()

    agent = get_agent(
        observation_space=observation_space, action_space=action_space, device=device, parameter=parameter
    )

    learner = Learner(agent=agent, queue=queue, device=device, parameter=parameter,)

    actors = [
        Actor(
            env_name=parameter.ENV_NAME, actor_id=actor_id, agent=agent,
            queue=queue, parameter=parameter
        ) for actor_id in range(parameter.N_ACTORS)
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

    print_basic_info(observation_space, action_space, device, parameter)


if __name__ == "__main__":
    assert parameter.AGENT_TYPE in OffPolicyAgentTypes
    main()
