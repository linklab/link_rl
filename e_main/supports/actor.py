from collections import deque

from gym.vector import AsyncVectorEnv
import torch.multiprocessing as mp

from b_environments.make_envs import make_gym_env
from g_utils.types import Transition


class Actor(mp.Process):
    def __init__(self, env_name, actor_id, agent, queue, params):
        super(Actor, self).__init__()
        self.env_name = env_name
        self.actor_id = actor_id
        self.agent = agent
        self.queue = queue
        self.params = params

        self.is_vectorized_env_created = mp.Value('i', False)
        self.is_terminated = mp.Value('i', False)

    def run(self):
        train_env = AsyncVectorEnv(
            env_fns=[
                make_gym_env(self.env_name) for _ in range(self.params.N_VECTORIZED_ENVS)
            ]
        )

        self.is_vectorized_env_created.value = True

        histories = []
        for _ in range(self.params.N_VECTORIZED_ENVS):
            histories.append(deque(maxlen=self.params.N_STEP))

        observations = train_env.reset()

        actor_time_step = 0

        while True:
            actor_time_step += 1
            actions = self.agent.get_action(observations)
            next_observations, rewards, dones, infos = train_env.step(actions)

            for env_id, (observation, action, next_observation, reward, done, info) in enumerate(
                    zip(observations, actions, next_observations, rewards, dones, infos)
            ):
                info["actor_id"] = self.actor_id
                info["env_id"] = env_id
                info["model_version_v"] = self.agent.model_version.value
                info["actor_time_step"] = actor_time_step
                histories[env_id].append(Transition(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    reward=reward,
                    done=done,
                    info=info
                ))

                if len(histories[env_id]) == self.params.N_STEP or done:
                    n_step_transitions = tuple(histories[env_id])
                    next_observation = n_step_transitions[-1].next_observation

                    n_step_reward = 0.0
                    for n_step_transition in reversed(n_step_transitions):
                        n_step_reward = n_step_transition.reward + \
                                        self.params.GAMMA * n_step_reward * \
                                        (0.0 if n_step_transition.done else 1.0)
                        if n_step_transition.done:
                            break

                    info["actor_id"] = self.actor_id
                    info["env_id"] = env_id
                    # NOTE: TODO 모든 스텝에 대하여 동일한 model_version_v 인지 체크
                    info["model_version_v"] = n_step_transitions[0].info["model_version_v"]
                    info["actor_time_step"] = n_step_transitions[0].info["actor_time_step"]
                    info["real_num_steps"] = len(n_step_transitions)
                    n_step_transition = Transition(
                        observation=n_step_transitions[0].observation,
                        action=n_step_transitions[0].action,
                        next_observation=next_observation,
                        reward=n_step_reward,
                        done=done,
                        info=info
                    )
                    self.queue.put(n_step_transition)

                    histories[env_id].clear()

            observations = next_observations

            if self.is_terminated.value:
                break

        self.queue.put(None)
