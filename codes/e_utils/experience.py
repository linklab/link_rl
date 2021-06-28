import os
import sys
import numpy as np
from collections import namedtuple, deque

from gym.vector import VectorEnv
from icecream import ic

from codes.c_models.continuous_action.continuous_action_model import ContinuousActionModel
from codes.e_utils.common_utils import map_range
from codes.e_utils.reward_changer import RewardChanger
from codes.e_utils import rl_utils

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

# one single experience step
from codes.d_agents.a0_base_agent import BaseAgent

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'done', 'info', 'agent_state', 'model_version')
)

ExperienceFirstLast = namedtuple(
    'ExperienceFirstLast',
    ('state', 'action', 'reward', 'last_state', 'last_step', 'done', 'info', 'agent_state', 'model_version')
)

class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """

    def __init__(self, env, agent, n_step=2, steps_delta=1):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param n_step: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        # assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(n_step, int)
        assert n_step >= 1

        assert isinstance(env, VectorEnv)

        self.env = env
        self.pool = [env]

        self.agent = agent
        self.n_step = n_step
        self.steps_delta = steps_delta
        self.episode_reward_lst = []
        self.episode_done_step_lst = []
        self.episode_idx = 0

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs = env.reset()
            self.episode_idx += 1
            # if the environment is vectorized, all it's output is lists of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            obs_len = len(obs)
            states.extend(obs)

            env_lens.append(obs_len)   # vectorized env는 self.pool 자체가 env이며 env_lens는 그 내부의 env 개수를 지니고 있음.

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.n_step))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(rl_utils.initial_agent_state())

        #ic(states, agent_states, histories, cur_rewards, cur_steps, env_lens)

        iter_idx = 0
        while True:
            actions = [None] * len(states)

            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].single_action_space.sample()  # assume that all envs are from the same family
                else:
                    states_input.append(state)
                    states_indices.append(idx)

            if states_input:
                new_actions, new_agent_states = self.agent(states_input, agent_states)
                for idx, action in enumerate(new_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]

            grouped_actions, grouped_agent_states = group_list(actions, agent_states, env_lens)

           #print(grouped_actions, grouped_agent_states, "!!!!!!")

            global_ofs = 0
            for env_idx, (env, action_n, agent_state) in enumerate(
                    zip(self.pool, grouped_actions, grouped_agent_states)
            ):
                action = np.asarray(action_n)

                if isinstance(self.agent.model, ContinuousActionModel):
                    action = map_range(
                        np.asarray(action),
                        np.ones_like(self.agent.action_min) * -1.0, np.ones_like(self.agent.action_max),
                        self.agent.action_min, self.agent.action_max
                    )

                next_state_n, r_n, is_done_n, info_n = env.step(action)

                #ic(env_idx, env, len(action_n), len(next_state_n), len(r_n), len(is_done_n), len(info_n))

                for ofs, (action, next_state, r, is_done, info) in enumerate(
                        zip(action_n, next_state_n, r_n, is_done_n, info_n)
                ):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    if isinstance(self.env.envs[0], RewardChanger):
                        cur_rewards[idx] += self.env.envs[0].reverse_reward(r)
                    else:
                        cur_rewards[idx] += r

                    cur_steps[idx] += 1

                    if state is not None:
                        history.append(Experience(
                            state=state, action=action, reward=r, done=is_done, info=info,
                            agent_state=agent_state,
                            model_version=self.agent.model_version if hasattr(self.agent, "model_version") else None
                        ))

                    if len(history) == self.n_step and iter_idx % self.steps_delta == 0:
                        yield tuple(history)

                    states[idx] = next_state

                    if is_done:
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.n_step:
                            yield tuple(history)

                        # generate tail of history
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)

                        if 'ale.lives' not in info or info['ale.lives'] == 0:
                            self.episode_reward_lst.append(cur_rewards[idx])
                            self.episode_done_step_lst.append(cur_steps[idx])
                            cur_rewards[idx] = 0.0
                            cur_steps[idx] = 0

                        # if params.NUM_ENVIRONMENTS == 1:
                        #     states[idx] = env.envs[0].reset()
                        # else:
                        #     states[idx] = None

                        states[idx] = None

                        # print("{0}, {1}, @@@@@".format(self.episode_idx, self.episode_reward_lst))
                        self.episode_idx += 1

                        # vectorized envs are reset automatically
                        # states[idx] = None
                        # print(self.episode_reward_lst, "@@@@@")

                        agent_states[idx] = rl_utils.initial_agent_state()
                        history.clear()

                global_ofs += len(action_n)
            iter_idx += 1

    def pop_episode_reward_lst(self):
        r = self.episode_reward_lst
        if r:
            self.episode_reward_lst = []
            self.episode_done_step_lst = []
        return r

    def pop_episode_reward_and_done_step_lst(self):
        res = self.episode_reward_lst, self.episode_done_step_lst
        if res:
            self.episode_reward_lst = []
            self.episode_done_step_lst = []
        return res


def group_list(actions, agent_states, env_lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    grouped_actions = []
    grouped_agent_states = []
    cur_ofs = 0
    for g_len in env_lens:
        grouped_actions.append(actions[cur_ofs: cur_ofs + g_len])
        grouped_agent_states.append(agent_states[cur_ofs: cur_ofs + g_len])
        cur_ofs += g_len
    return grouped_actions, grouped_agent_states


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, n_step=1, steps_delta=1):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, n_step + 1, steps_delta)
        self.gamma = gamma
        self.n_step_ = n_step

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():

            #print(exp)

            if exp[-1].done and len(exp) <= self.n_step_:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]

            total_reward = 0.0
            for e in reversed(elems):
                total_reward = e.reward + self.gamma * total_reward

            e = ExperienceFirstLast(
                state=exp[0].state, action=exp[0].action, reward=total_reward, last_state=last_state,
                done=exp[0].done, info=exp[0].info, last_step=len(elems),
                agent_state=exp[0].agent_state,
                model_version=self.agent.model_version if hasattr(self.agent, "model_version") else None
            )

            #print(e)

            yield e
