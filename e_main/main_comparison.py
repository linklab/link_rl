import os
import sys

from g_utils.commons import print_basic_info, get_agents

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
))

from e_main.supports.main_preamble import *
from a_configuration.parameter_comparison import ParameterComparison as params_c


def main():
    wandb_configuration = {
        key: getattr(params_c, key) for key in dir(params_c) if not key.startswith("__")
    }
    if params_c.USE_WANDB:
        wandb_obj = wandb.init(
            entity=params_c.WANDB_ENTITY,
            project="{0}_{1}".format(
                params_c.ENV_NAME, "_".join(*params_c.AGENTS)
            ),
            config=wandb_configuration
        )
    else:
        wandb_obj = None

    num_agents = len(params_c.AGENTS)

    test_envs = [gym.make(params_c.ENV_NAME) for _ in range(num_agents)]
    n_features = test_envs[0].observation_space.shape[0]
    n_actions = test_envs[0].action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    queue = mp.Queue()

    print_basic_info(device, params_c)
    print_params(params_c)

    agents = get_agents(n_features, n_actions, device, params_c)

    actors = [
        Actor(
            env_name=params_c.ENV_NAME,
            actor_id=actor_id,
            agent=agent,
            queue=queue,
            params=params
        ) for actor_id in range(params.N_ACTORS)
    ]

    learner = Learner(
        test_env=test_env,
        agent=agent,
        queue=queue,
        device=device,
        params=params
    )

    for actor in actors:
        actor.start()

    # Busy Wait: 모든 액터들이 VecEnv를 생성 완료할 때까지 대기
    for actor in actors:
        while not actor.is_vectorized_env_created.value:
            time.sleep(0.1)

    print("########## LEARNING STARTED !!! ##########")

    learner.start()

    while True:
        if params.USE_WANDB:
            wandb_log(agent, learner, wandb_obj, params)

        # Busy Wait: learner에서 학습 완료될 때까지 대기
        if learner.is_terminated.value:
            # learner가 학습 완료하면 각 actor들의 rollout 종료
            for actor in actors:
                actor.is_terminated.value = True

            break

        time.sleep(0.5)

    # Busy Wait: 모든 actor가 조인할 때까지 대기
    for actor in actors:
        while actor.is_alive():
            actor.join(timeout=1)

    # Busy Wait: learner가 조인할 때까지 대기
    while learner.is_alive():
        if params.USE_WANDB:
            wandb_log(agent, learner, wandb_obj)
        learner.join(timeout=1)


if __name__ == "__main__":
    main()
