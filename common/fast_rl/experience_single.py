import numpy as np

from collections import namedtuple, deque

from common.fast_rl.experience import ExperienceWithNoise, Experience, ExperienceFirstLastWithNoise, ExperienceFirstLast
from .rl_agent import BaseAgent, AgentDDPG, AgentD4PG


class ExperienceSourceSingleEnv:
    """
    Simple n-step experience source using only SINGLE environment
    Every experience contains n list of Experience entries
    """

    def __init__(self, env, agent, steps_count=2, step_length=-1, render=False):
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        self.env = env
        self.agent = agent
        self.steps_count = steps_count
        self.step_length = step_length  # -1 이면 MLP or CNN, 1 이상의 값이면 RNN
        self.render = render
        self.episode_reward_lst = []
        self.episode_done_step_lst = []
        self.state_deque = deque(maxlen=30)

    def get_processed_state(self, new_state):
        self.state_deque.append(new_state)

        if self.step_length == -1:
            next_state = np.array(self.state_deque[-1])
        elif self.step_length >= 1:
            if len(self.state_deque) < self.step_length:
                next_state = list(self.state_deque)

                for _ in range(self.step_length - len(self.state_deque)):
                    next_state.insert(0, np.zeros(shape=self.env.observation_space.shape))

                next_state = np.array(next_state)
            else:
                next_state = np.array(
                    [
                        self.state_deque[-self.step_length + offset] for offset in range(self.step_length)
                    ]
                )
        else:
            raise ValueError()

        return next_state

    def __iter__(self):
        state = self.env.reset()

        history = deque(maxlen=self.steps_count)
        cur_reward = 0.0
        cur_step = 0
        agent_state = self.agent.initial_agent_state()
        noise = None

        iter_idx = 0
        while True:
            if self.render:
                self.env.render()

            states_input = []

            processed_state = self.get_processed_state(state)

            states_input.append(processed_state)

            agent_states_input = []
            agent_states_input.append(agent_state)
            if isinstance(self.agent, AgentDDPG):
                actions, noises, new_agent_states = self.agent(states_input, agent_states_input)
            else:
                actions, new_agent_states = self.agent(states_input, agent_states_input)

            agent_state = new_agent_states[0]
            action = actions[0]

            next_state, r, is_done, info = self.env.step(action)

            if 'original_reward' in info:
                cur_reward += info['original_reward']
            else:
                cur_reward += r

            cur_step += 1

            if state is not None:
                if isinstance(self.agent, AgentDDPG):
                    history.append(ExperienceWithNoise(
                        state=processed_state, action=action, noise=noise, reward=r, done=is_done, info=info
                    ))
                else:
                    history.append(Experience(
                        state=processed_state, action=action, reward=r, done=is_done, info=info
                    ))

            if len(history) == self.steps_count:
                yield tuple(history)

            state = next_state

            if is_done:
                # in case of very short episode (shorter than our steps count), send gathered history
                if 0 < len(history) < self.steps_count:
                    yield tuple(history)

                # generate tail of history
                while len(history) > 1:
                    # removes the element (the old one) from the left side of the deque and returns the value
                    history.popleft()
                    yield tuple(history)

                state = self.env.reset()

                agent_state = self.agent.initial_agent_state()

                if 'ale.lives' not in info or info['ale.lives'] == 0:
                    self.episode_reward_lst.append(cur_reward)
                    self.episode_done_step_lst.append(cur_step)
                    cur_reward = 0.0
                    cur_step = 0

                history.clear()

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


class ExperienceSourceSingleEnvFirstLast(ExperienceSourceSingleEnv):
    def __init__(self, env, agent, gamma, steps_count=1, step_length=-1, render=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceSingleEnvFirstLast, self).__init__(env, agent, steps_count + 1, step_length, render)
        self.gamma = gamma
        self.steps_count = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceSingleEnvFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps_count:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward

            if isinstance(self.agent, AgentDDPG) or isinstance(self.agent, AgentD4PG):
                exp = ExperienceFirstLastWithNoise(
                    state=exp[0].state, action=exp[0].action, noise=exp[0].noise, reward=total_reward,
                    last_state=last_state, last_step=len(elems), done=exp[-1].done, info=exp[-1].info
                )
            else:
                exp = ExperienceFirstLast(
                    state=exp[0].state, action=exp[0].action, reward=total_reward, last_state=last_state,
                    last_step=len(elems), done=exp[-1].done, info=exp[-1].info
                )
            yield exp