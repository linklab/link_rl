import gym
import torch

import numpy as np

from collections import namedtuple, deque

from .rl_agent import BaseAgent, AgentDDPG, AgentD4PG

# one single experience step
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'done', 'info', 'model_version']
)

ExperienceFirstLast = namedtuple(
    'ExperienceFirstLast',
    ('state', 'action', 'reward', 'last_state', 'last_step', 'done', 'info', 'model_version')
)


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """

    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        # assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.episode_reward_lst = []
        self.episode_done_step_lst = []
        self.vectorized = vectorized

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)

            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_agent_state())

        # print(states, agent_states, histories, cur_rewards, cur_steps)

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                if isinstance(self.agent, AgentDDPG) or isinstance(self.agent, AgentD4PG):
                    states_actions, noises, new_agent_states = self.agent(states_input, agent_states)
                else:
                    states_actions, new_agent_states = self.agent(states_input, agent_states)

                # TODO: noise??? ?????? ??????

                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]

            grouped_actions = _group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, is_done_n, info = env.step(action_n)
                else:
                    next_state, r, is_done, info = env.step(action_n[0])
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    if 'original_reward' in info:
                        cur_rewards[idx] += info['original_reward']
                    else:
                        cur_rewards[idx] += r

                    cur_steps[idx] += 1

                    if state is not None:
                        history.append(Experience(
                            state=state, action=action, reward=r, done=is_done, info=info,
                            model_version=self.agent.model_version if hasattr(self.agent, "model_version") else None
                        ))

                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state

                    if is_done:
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)

                        # generate tail of history
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)

                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_agent_state()

                        if 'ale.lives' not in info or info['ale.lives'] == 0:
                            self.episode_reward_lst.append(cur_rewards[idx])
                            self.episode_done_step_lst.append(cur_steps[idx])
                            cur_rewards[idx] = 0.0
                            cur_steps[idx] = 0
                        #     # vectorized envs are reset automatically
                        #     states[idx] = env.reset() if not self.vectorized else None
                        #     agent_states[idx] = self.agent.initial_agent_state()

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
        res = list(zip(self.episode_reward_lst, self.episode_done_step_lst))
        if res:
            self.episode_reward_lst = []
            self.episode_done_step_lst = []
        return res


def _group_list(items, lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs + g_len])
        cur_ofs += g_len
    return res


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """

    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count + 1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(
                state=exp[0].state, action=exp[0].action, reward=total_reward, last_state=last_state,
                last_step=len(elems), info=exp[0].info, done=exp[0].done,
                model_version=self.agent.model_version if hasattr(self.agent, "model_version") else None
            )


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


class ExperienceSourceRollouts:
    """
    N-step rollout experience source following A3C rollouts scheme. Have to be used with agent,
    keeping the value in its state (for example, agent.ActorCriticAgent).

    Yields batches of num_envs * n_steps samples with the following arrays:
    1. observations
    2. actions
    3. discounted rewards, with values approximation
    4. values
    """

    def __init__(self, env, agent, gamma, steps_count=5):
        """
        Constructs the rollout experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions
        :param steps_count: how many steps to perform rollouts
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(gamma, float)
        assert isinstance(steps_count, int)
        assert steps_count >= 1

        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        self.episode_reward_lst = []
        self.episode_done_step_lst = []

    def __iter__(self):
        pool_size = len(self.pool)
        states = [np.array(e.reset()) for e in self.pool]
        mb_states = np.zeros((pool_size, self.steps_count) + states[0].shape, dtype=states[0].dtype)
        mb_rewards = np.zeros((pool_size, self.steps_count), dtype=np.float32)
        mb_values = np.zeros((pool_size, self.steps_count), dtype=np.float32)
        mb_actions = np.zeros((pool_size, self.steps_count), dtype=np.int64)
        mb_dones = np.zeros((pool_size, self.steps_count), dtype=np.bool)
        episode_reward_lst = [0.0] * pool_size
        episode_done_step_lst = [0] * pool_size
        critics = None
        step_idx = 0

        while True:
            if isinstance(self.agent, AgentDDPG):
                actions, noise, critics = self.agent(states, critics)
            else:
                actions, critics = self.agent(states, critics)

            # TODO: noise??? ?????? ??????

            rewards = []
            dones = []
            new_states = []
            for env_idx, (e, action) in enumerate(zip(self.pool, actions)):
                o, r, done, _ = e.step(action)
                episode_reward_lst[env_idx] += r
                episode_done_step_lst[env_idx] += 1
                if done:
                    o = e.reset()
                    self.episode_reward_lst.append(episode_reward_lst[env_idx])
                    self.episode_done_step_lst.append(episode_done_step_lst[env_idx])
                    episode_reward_lst[env_idx] = 0.0
                    episode_done_step_lst[env_idx] = 0
                new_states.append(np.array(o))
                dones.append(done)
                rewards.append(r)

            # we need an extra step to get values approximation for rollouts
            if step_idx == self.steps_count:
                print(mb_rewards, mb_dones, critics)
                # calculate rollout rewards
                for env_idx, (env_rewards, env_dones, last_value) in enumerate(zip(mb_rewards, mb_dones, critics)):
                    env_rewards = env_rewards.tolist()
                    env_dones = env_dones.tolist()
                    if not env_dones[-1]:
                        env_rewards = discount_with_dones(env_rewards + [last_value], env_dones + [False], self.gamma)[
                                      :-1]
                    else:
                        env_rewards = discount_with_dones(env_rewards, env_dones, self.gamma)
                    mb_rewards[env_idx] = env_rewards
                yield mb_states.reshape(
                    (-1,) + mb_states.shape[2:]), mb_rewards.flatten(), mb_actions.flatten(), mb_values.flatten()
                step_idx = 0

            mb_states[:, step_idx] = states
            mb_rewards[:, step_idx] = rewards
            mb_values[:, step_idx] = critics
            mb_actions[:, step_idx] = actions
            mb_dones[:, step_idx] = dones
            step_idx += 1
            states = new_states

    def pop_episode_reward_lst(self):
        r = self.episode_reward_lst
        if r:
            self.episode_reward_lst = []
            self.episode_done_step_lst = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.episode_reward_lst, self.episode_done_step_lst))
        if res:
            self.episode_reward_lst, self.episode_done_step_lst = [], []
        return res


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
