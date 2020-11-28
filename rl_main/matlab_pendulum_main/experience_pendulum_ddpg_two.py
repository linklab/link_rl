import collections
import enum
import time
import numpy as np

from collections import namedtuple, deque

# one single experience step
from common.environments.matlab.matlabenv import Status
from common.fast_rl.common.statistics import StatisticsForValueBasedRL, StatisticsForPolicyBasedRL
from common.fast_rl.rl_agent import BaseAgent, AgentDDPG
from config.names import DeepLearningModelName

ExperienceWithNoise = namedtuple('ExperienceWithNoise', ['state', 'action', 'noise', 'reward', 'done', 'agent_type'])

ExperienceFirstLastWithNoise = collections.namedtuple(
    'ExperienceFirstLastWithNoise', ['state', 'action', 'noise', 'reward', 'last_state', 'last_step', 'done', 'agent_type']
)


class AgentType(enum.Enum):
    SWING_UP_AGENT = 0,
    BALANCING_AGENT = 1


class ExperienceSourceSingleEnvDdpgTwo:
    """
    Simple n-step experience source using only SINGLE environment
    Every experience contains n list of Experience entries
    """

    def __init__(self, params, env, agent_swing_up, agent_balancing, steps_count=2, step_length=-1, render=False):
        assert isinstance(agent_swing_up, BaseAgent)
        assert isinstance(agent_balancing, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        self.params = params
        self.env = env
        self.agent_swing_up = agent_swing_up
        self.agent_balancing = agent_balancing
        self.steps_count = steps_count
        self.step_length = step_length  # -1 이면 MLP, 1 이상의 값이면 RNN
        self.render = render

        self.episode_reward_and_info_lst = []

        self.episode_continuous_positive_actions = []
        self.episode_continuous_negative_actions = []
        self.state_deque = deque(maxlen=30)

        self.current_agent = self.agent_swing_up
        self.current_agent_type = AgentType.SWING_UP_AGENT

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

    def set_current_agent(self, state):
        status_value = state[-1]

        if status_value in [Status.SWING_UP.value, Status.SWING_UP_TO_BALANCING]:
            self.current_agent = self.agent_swing_up
            self.current_agent_type = AgentType.SWING_UP_AGENT
        else:
            self.current_agent = self.agent_balancing
            self.current_agent_type = AgentType.BALANCING_AGENT

    def __iter__(self):
        state = self.env.reset()
        self.set_current_agent(state)

        history = deque(maxlen=self.steps_count)
        cur_reward = 0.0
        cur_step = 0
        agent_state = self.current_agent.initial_agent_state()

        iter_idx = 0
        while True:
            if self.render:
                self.env.render()

            states_input = []

            processed_state = self.get_processed_state(state)

            states_input.append(processed_state)

            agent_states_input = []
            agent_states_input.append(agent_state)
            actions, noises, new_agent_states = self.current_agent(states_input, agent_states_input)
            noise = noises[0]

            agent_state = new_agent_states[0]
            action = actions[0]

            next_state, r, is_done, info = self.env.step(action)

            cur_reward += r
            cur_step += 1

            if state is not None:
                history.append(
                    ExperienceWithNoise(
                        state=processed_state,
                        action=action,
                        noise=noise,
                        reward=r,
                        done=is_done,
                        agent_type=self.current_agent_type
                    )
                )

            if len(history) == self.steps_count:
                yield tuple(history)

            state = next_state
            self.set_current_agent(state)

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
                self.set_current_agent(state)

                agent_state = self.current_agent.initial_agent_state()

                self.episode_reward_and_info_lst.append((cur_reward, info))

                cur_reward = 0.0
                cur_step = 0

                history.clear()

            iter_idx += 1

    def pop_episode_reward_and_info_lst(self):
        episode_reward_and_info = self.episode_reward_and_info_lst
        if episode_reward_and_info:
            self.episode_reward_and_info_lst = []
        return episode_reward_and_info


class ExperienceSourceSingleEnvFirstLastDdpgTwo(ExperienceSourceSingleEnvDdpgTwo):
    def __init__(self, params, env, agent_swing_up, agent_balancing, gamma, steps_count=1, step_length=-1, render=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceSingleEnvFirstLastDdpgTwo, self).__init__(
            params, env, agent_swing_up, agent_balancing, steps_count + 1, step_length, render
        )
        self.gamma = gamma
        self.steps_count = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceSingleEnvFirstLastDdpgTwo, self).__iter__():
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

            exp = ExperienceFirstLastWithNoise(
                state=exp[0].state, action=exp[0].action, noise=exp[0].noise, reward=total_reward,
                last_state=last_state, last_step=len(elems), done=exp[-1].done, agent_type=self.current_agent_type
            )

            yield exp


class RewardTrackerMatlabPendulum:
    def __init__(self, params, stop_mean_episode_reward, average_size_for_stats, frame=True, draw_viz=True, stat=None):
        self.params = params
        self.min_ts_diff = 1    # 1 second
        self.stop_mean_episode_reward = stop_mean_episode_reward
        self.stat = stat
        self.average_size_for_stats = average_size_for_stats
        self.draw_viz = draw_viz
        self.frame = frame
        self.episode_reward_list = None
        self.done_episodes = 0
        self.mean_episode_reward = 0.0

    def __enter__(self):
        self.start_ts = time.time()
        self.ts = time.time()
        self.ts_frame = 0
        self.episode_reward_list = []
        return self

    def start_reward_track(self):
        self.__enter__()

    def __exit__(self, *args):
        pass

    def set_episode_reward(self, episode_reward_and_info, episode_done_step, epsilon, action_count=None, continuous_action_mean=None):
        assert not (action_count and continuous_action_mean)
        self.done_episodes += 1
        self.episode_reward_list.append(episode_reward_and_info[0])
        episode_info = episode_reward_and_info[1]
        self.mean_episode_reward = np.mean(self.episode_reward_list[-self.average_size_for_stats:])

        current_ts = time.time()
        elapsed_time = current_ts - self.start_ts
        ts_diff = current_ts - self.ts

        is_print_performance = False

        if ts_diff > self.min_ts_diff:
            is_print_performance = True
            self.print_performance(
                episode_done_step, current_ts, ts_diff, self.mean_episode_reward,
                epsilon, elapsed_time, action_count, episode_info
            )

        if self.mean_episode_reward > self.stop_mean_episode_reward:
            if not is_print_performance:
                self.print_performance(
                    episode_done_step, current_ts, ts_diff, self.mean_episode_reward,
                    epsilon, elapsed_time, action_count, episode_info
                )
            if self.frame:
                print("Solved in {0} frames and {1} episodes!".format(episode_done_step, self.done_episodes))
            else:
                print("Solved in {0} steps and {1} episodes!".format(episode_done_step, self.done_episodes))
            return True, self.mean_episode_reward

        return False, self.mean_episode_reward

    def print_performance(
            self, episode_done_step, current_ts, ts_diff, mean_episode_reward, epsilon, elapsed_time, action_count, episode_info
    ):
        speed = (episode_done_step - self.ts_frame) / ts_diff
        self.ts_frame = episode_done_step
        self.ts = current_ts

        if isinstance(epsilon, tuple) or isinstance(epsilon, list):
            epsilon_str = "{0:5.3f}, {1:5.3f}".format(
                epsilon[0] if epsilon[0] else 0.0,
                epsilon[1] if epsilon[1] else 0.0
            )
        else:
            epsilon_str = "{0:5.3f}".format(
                epsilon if epsilon else 0.0,
            )

        episode_reward_str = "{0:6.1f} [{1:6.1f}, {2:6.1f}, {3:6.1f}]".format(
            self.episode_reward_list[-1],
            episode_info["episode_position_reward_list"],
            episode_info["episode_pendulum_velocity_reward"],
            episode_info["episode_action_reward"]
        )

        print(
            "[{0:6}/{1}] done {2:4} games, episode_reward: {3}, mean_{4}_episode_reward: {5:7.3f}, "
            "status: [{6:3d}|{7:3d}], epsilon: {8}, speed: {9:7.2f} {8}, elapsed time: {10}".format(
                episode_done_step,
                self.params.MAX_GLOBAL_STEPS,
                len(self.episode_reward_list),
                episode_reward_str,
                self.average_size_for_stats,
                mean_episode_reward,
                episode_info["count_swing_up_states"],
                episode_info["count_balancing_states"],
                epsilon_str,
                speed,
                "fps" if self.frame else "steps/sec.",
                time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
        ), end="")

        if action_count:
            print(", {0}".format(action_count), flush=True)
        else:
            print("", flush=True)

        if self.draw_viz and self.stat:
            if isinstance(self.stat, StatisticsForValueBasedRL):
                self.stat.draw_performance(episode_done_step, mean_episode_reward, speed, epsilon)
            elif isinstance(self.stat, StatisticsForPolicyBasedRL):
                self.stat.draw_performance(episode_done_step, mean_episode_reward, speed)
            else:
                raise ValueError()