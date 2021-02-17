import os
import sys
from collections import namedtuple

import gym
import numpy as np
from gym.vector import VectorEnv

from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils.experience import ExperienceFirstLast

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.d_agents.a0_base_agent import BaseAgent

ExperienceFirstLastRollout = namedtuple(
    'ExperienceFirstLastRollout', ('state', 'action', 'reward', 'value')
)

def discount_return_with_dones(rewards, dones, gamma):
    discounted_return = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)
        discounted_return.append(r)
    return discounted_return[::-1]


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

    def __init__(self, env, agent, gamma, n_step=5):
        """
        Constructs the rollout experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions
        :param n_step: how many steps to perform rollouts
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, OnPolicyAgent)
        assert isinstance(gamma, float)
        assert isinstance(n_step, int)
        assert n_step >= 1

        assert isinstance(env, VectorEnv)

        self.pool = [env]
        self.agent = agent
        self.gamma = gamma
        self.n_step = n_step
        self.episode_reward_lst = []
        self.episode_done_step_lst = []

    def __iter__(self):
        pool_size = len(self.pool)
        states = [np.array(e.reset()) for e in self.pool]
        mb_states = np.zeros((pool_size, self.n_step) + states[0].shape, dtype=states[0].dtype)
        mb_rewards = np.zeros((pool_size, self.n_step), dtype=np.float32)
        mb_values = np.zeros((pool_size, self.n_step), dtype=np.float32)
        mb_actions = np.zeros((pool_size, self.n_step), dtype=np.int64)
        mb_dones = np.zeros((pool_size, self.n_step), dtype=np.bool)
        episode_reward_lst = [0.0] * pool_size
        episode_done_step_lst = [0] * pool_size
        critics = None
        step_idx = 0

        while True:
            actions, critics = self.agent(states, critics)
            rewards = []
            dones = []
            new_states = []
            for env_idx, (env, action) in enumerate(zip(self.pool, actions)):
                next_state, r, done, info = env.step(action)
                episode_reward_lst[env_idx] += r
                episode_done_step_lst[env_idx] += 1
                if done:
                    next_state = env.reset()
                    self.episode_reward_lst.append(episode_reward_lst[env_idx])
                    self.episode_done_step_lst.append(episode_done_step_lst[env_idx])
                    # episode_reward_lst[env_idx] = 0.0
                    episode_done_step_lst[env_idx] = 0
                new_states.append(np.array(next_state))
                dones.append(done)
                rewards.append(r)

            # we need an extra step to get values approximation for rollouts
            if step_idx == self.n_step:
                # print(mb_rewards, mb_dones, critics)
                # calculate rollout rewards
                for env_idx, (env_rewards, env_dones, last_value) in enumerate(zip(mb_rewards, mb_dones, critics)):
                    env_rewards = env_rewards.tolist()
                    env_dones = env_dones.tolist()
                    if not env_dones[-1]:
                        env_rewards = discount_return_with_dones(
                            env_rewards + [last_value], env_dones + [False], self.gamma
                        )[:-1]
                    else:
                        env_rewards = discount_return_with_dones(env_rewards, env_dones, self.gamma)
                    mb_rewards[env_idx] = env_rewards

                exp = ExperienceFirstLastRollout(
                    state=mb_states.reshape((-1,) + mb_states.shape[2:]),
                    action=mb_actions.flatten(),
                    reward=mb_rewards.flatten(),
                    value=mb_values.flatten()
                    # last_state=last_state,
                    # last_step=len(elems),
                    # info=exp[0].info,
                    # done=exp[0].done
                )

                print(exp.state.shape, exp.action.shape, exp.reward.shape, exp.value.shape)
                yield exp
                # yield mb_states.reshape((-1,) + mb_states.shape[2:]), mb_rewards.flatten(), mb_actions.flatten(), \
                #       mb_values.flatten()
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

    def pop_episode_reward_and_done_step_lst(self):
        res = self.episode_reward_lst, self.episode_done_step_lst
        if res:
            self.episode_reward_lst = []
            self.episode_done_step_lst = []
        return res

    def pop_rewards_steps(self):
        res = list(zip(self.episode_reward_lst, self.episode_done_step_lst))
        if res:
            self.episode_reward_lst, self.episode_done_step_lst = [], []
        return res