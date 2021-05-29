import math
import os
import sys
import gym
import torch
import numpy as np
from collections import namedtuple, deque

from gym.vector import VectorEnv
from icecream import ic

from codes.e_utils.names import EnvironmentName, RLAlgorithmName
from codes.e_utils.reward_changer import RewardChanger

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

# one single experience step
from codes.d_agents.a0_base_agent import BaseAgent
from codes.a_config.parameters import PARAMETERS as params

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'info'])
ExperienceWithNoise = namedtuple(
    'ExperienceWithNoise', ['state', 'action', 'noise', 'reward', 'done', 'info']
)

ExperienceFirstLast = namedtuple(
    'ExperienceFirstLast',
    ('state', 'action', 'reward', 'last_state', 'last_step', 'done', 'info')
)
ExperienceFirstLastWithNoise = namedtuple(
    'ExperienceFirstLastWithNoise',
    ('state', 'action', 'noise', 'reward', 'last_state', 'last_step', 'done', 'info')
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

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            obs_len = len(obs)
            states.extend(obs)

            env_lens.append(obs_len)   # vectorized env는 self.pool 자체가 env이며 env_lens는 그 내부의 env 개수를 지니고 있음.

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.n_step))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_agent_state())

        ic(states, agent_states, histories, cur_rewards, cur_steps, env_lens)

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            agent_states_input = []
            for idx, (state, agent_state) in enumerate(zip(states, agent_states)):
                if state is None:
                    actions[idx] = self.pool[0].single_action_space.sample()  # assume that all envs are from the same family
                else:
                    states_input.append(state)
                    agent_states_input.append(agent_state)
                    states_indices.append(idx)
                    #ic(agent_states_input, "@@")

            #ic(states_input, agent_states_input)

            if states_input:
                new_actions, new_agent_states = self.agent(states_input, agent_states_input)
                for idx, action in enumerate(new_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            else:
                pass

            grouped_actions = _group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                action = np.asarray(action_n)

                if params.RL_ALGORITHM in [RLAlgorithmName.DDPG_V0, RLAlgorithmName.TD3_V0, RLAlgorithmName.SAC_V0]:
                    if hasattr(self.agent.params, "ACTION_SCALE") and self.agent.params.ACTION_SCALE:
                        action = self.agent.params.ACTION_SCALE * action

                next_state_n, r_n, is_done_n, info_n = env.step(action)

                #ic(env_idx, env, len(action_n), len(next_state_n), len(r_n), len(is_done_n), len(info_n))

                for ofs, (action, next_state, r, is_done, info) in enumerate(zip(action_n, next_state_n, r_n, is_done_n, info_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    if isinstance(self.env.envs[0], RewardChanger):
                        cur_rewards[idx] += self.env.envs[0].reverse_reward(r)
                    else:
                        cur_rewards[idx] += r

                    cur_steps[idx] += 1

                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=is_done, info=info))

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

                        # vectorized envs are reset automatically
                        if params.NUM_ENVIRONMENTS == 1:
                            states[idx] = env.envs[0].reset()
                        else:
                            states[idx] = None

                        # states[idx] = env.reset()

                        agent_states[idx] = self.agent.initial_agent_state()
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


def _group_list(actions, env_lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in env_lens:
        res.append(actions[cur_ofs: cur_ofs + g_len])
        cur_ofs += g_len
    return res


class ExperienceSourceNamedTuple(ExperienceSource):
    """
    convert tuple to namedtuple
    """

    def __init__(self, env, agent, n_step=2, steps_delta=1):
        super(ExperienceSourceNamedTuple, self).__init__(env, agent, n_step, steps_delta)

    def __iter__(self):
        for exp in super(ExperienceSourceNamedTuple, self).__iter__():
            yield Experience(
                state=exp[0].state, action=exp[0].action, reward=exp[0].reward, done=exp[0].done
            )


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, n_step=1, steps_delta=1, vectorized=True):
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
                last_step=len(elems), info=exp[0].info, done=exp[0].done
            )

            #print(e)

            yield e


class BatchPreprocessor:
    """
    Abstract preprocessor class descendants to which converts experience
    batch to form suitable to learning.
    """

    def preprocess(self, batch):
        raise NotImplementedError


class QLearningPreprocessor(BatchPreprocessor):
    """
    Supports SimpleDQN, TargetDQN, DoubleDQN and can additionally feed TD-error back to
    experience replay buffer.

    To use different modes, use appropriate class method
    """

    def __init__(self, model, target_model, use_double_dqn=False, batch_td_error_hook=None, gamma=0.99, device="cpu"):
        self.model = model
        self.target_model = target_model
        self.use_double_dqn = use_double_dqn
        self.batch_dt_error_hook = batch_td_error_hook
        self.gamma = gamma
        self.device = device

    @staticmethod
    def simple_dqn(model, **kwargs):
        return QLearningPreprocessor(model=model, target_model=None, use_double_dqn=False, **kwargs)

    @staticmethod
    def target_dqn(model, target_model, **kwards):
        return QLearningPreprocessor(model, target_model, use_double_dqn=False, **kwards)

    @staticmethod
    def double_dqn(model, target_model, **kwargs):
        return QLearningPreprocessor(model, target_model, use_double_dqn=True, **kwargs)

    def _calc_Q(self, states_first, states_last):
        """
        Calculates apropriate q values for first and last states. Way of calculate depends on our settings.
        :param states_first: numpy array of first states
        :param states_last: numpy array of last states
        :return: tuple of numpy arrays of q values
        """
        # here we need both first and last values calculated using our main model, so we
        # combine both states into one batch for efficiency and separate results later
        if self.target_model is None or self.use_double_dqn:
            states_t = torch.tensor(np.concatenate((states_first, states_last), axis=0)).to(self.device)
            res_both = self.model(states_t).data.cpu().numpy()
            return res_both[:len(states_first)], res_both[len(states_first):]

        # in this case we have target_model set and use_double_dqn==False
        # so, we should calculate first_q and last_q using different models
        states_first_v = torch.tensor(states_first).to(self.device)
        states_last_v = torch.tensor(states_last).to(self.device)
        q_first = self.model(states_first_v).data
        q_last = self.target_model(states_last_v).data
        return q_first.cpu().numpy(), q_last.cpu().numpy()

    def _calc_target_rewards(self, states_last, q_last):
        """
        Calculate rewards from final states according to variants from our construction:
        1. simple DQN: max(Q(states, model))
        2. target DQN: max(Q(states, target_model))
        3. double DQN: Q(states, target_model)[argmax(Q(states, model)]
        :param states_last: numpy array of last states from the games
        :param q_last: numpy array of last q values
        :return: vector of target rewards
        """
        # in this case we handle both simple DQN and target DQN
        if self.target_model is None or not self.use_double_dqn:
            return q_last.max(axis=1)

        # here we have target_model set and use_double_dqn==True
        actions = q_last.argmax(axis=1)
        # calculate Q values using target net
        states_last_v = torch.tensor(states_last).to(self.device)
        q_last_target = self.target_model(states_last_v).data.cpu().numpy()
        return q_last_target[range(q_last_target.shape[0]), actions]

    def preprocess(self, batch):
        """
        Calculates data for Q learning from batch of observations
        :param batch: list of lists of Experience objects
        :return: tuple of numpy arrays:
            1. states -- observations
            2. target Q-values
            3. vector of td errors for every batch entry
        """
        # first and last states for every entry
        state_0 = np.array([exp[0].state for exp in batch], dtype=np.float32)
        state_L = np.array([exp[-1].state for exp in batch], dtype=np.float32)

        q0, qL = self._calc_Q(state_0, state_L)
        rewards = self._calc_target_rewards(state_L, qL)

        td = np.zeros(shape=(len(batch),))

        for idx, (total_reward, exps) in enumerate(zip(rewards, batch)):
            # game is done, no final reward
            if exps[-1].done:
                total_reward = 0.0
            for exp in reversed(exps[:-1]):
                total_reward *= self.gamma
                total_reward += exp.reward
            # update total reward and calculate td error
            act = exps[0].action
            td[idx] = q0[idx][act] - total_reward
            q0[idx][act] = total_reward

        return state_0, q0, td
