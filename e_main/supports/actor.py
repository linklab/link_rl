import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

from collections import deque
import torch.multiprocessing as mp

from g_utils.commons import get_train_env
from g_utils.types import Transition


class Actor(mp.Process):
    def __init__(self, env_name, actor_id, agent, queue, parameter):
        super(Actor, self).__init__()
        self.env_name = env_name
        self.actor_id = actor_id
        self.agent = agent
        self.queue = queue
        self.parameter = parameter

        self.is_vectorized_env_created = mp.Value('i', False)
        self.is_terminated = mp.Value('i', False)

        self.train_env = None

        self.histories = None

    def run(self):
        self.train_env = get_train_env(self.parameter)

        self.is_vectorized_env_created.value = True

        self.histories = []
        for _ in range(self.parameter.N_VECTORIZED_ENVS):
            self.histories.append(deque(maxlen=self.parameter.N_STEP))

        self.roll_out()

    def roll_out(self):
        observations = self.train_env.reset()

        actor_time_step = 0

        while True:
            actor_time_step += 1
            actions = self.agent.get_action(observations)
            next_observations, rewards, dones, infos = self.train_env.step(actions)

            for env_id, (observation, action, next_observation, reward, done, info) in enumerate(
                    zip(observations, actions, next_observations, rewards, dones, infos)
            ):
                info["actor_id"] = self.actor_id
                info["env_id"] = env_id
                info["actor_time_step"] = actor_time_step
                self.histories[env_id].append(Transition(
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    reward=reward,
                    done=done,
                    info=info
                ))

                if len(self.histories[env_id]) == self.parameter.N_STEP or done:
                    n_step_transition = Actor.get_n_step_transition(
                        history=self.histories[env_id], env_id=env_id,
                        actor_id=self.actor_id, info=info, done=done,
                        parameter=self.parameter
                    )
                    self.queue.put(n_step_transition)

            observations = next_observations

            if self.is_terminated.value:
                break

        self.queue.put(None)
        #
        # if self.parameter.AGENT_TYPE in OffPolicyAgentTypes:
        #     self.queue.put(None)
        # elif self.parameter.AGENT_TYPE in OnPolicyAgentTypes:
        #     yield None
        # else:
        #     raise ValueError()

    @staticmethod
    def get_n_step_transition(history, env_id, actor_id, info, done, parameter):
        n_step_transitions = tuple(history)
        next_observation = n_step_transitions[-1].next_observation

        n_step_reward = 0.0
        for n_step_transition in reversed(n_step_transitions):
            n_step_reward = n_step_transition.reward + \
                            parameter.GAMMA * n_step_reward * \
                            (0.0 if n_step_transition.done else 1.0)
            if n_step_transition.done:
                break

        info["actor_id"] = actor_id
        info["env_id"] = env_id
        info["actor_time_step"] = n_step_transitions[0].info["actor_time_step"]
        info["real_n_steps"] = len(n_step_transitions)
        n_step_transition = Transition(
            observation=n_step_transitions[0].observation,
            action=n_step_transitions[0].action,
            next_observation=next_observation,
            reward=n_step_reward,
            done=done,
            info=info
        )

        history.clear()

        return n_step_transition
