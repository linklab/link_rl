import random
import numpy as np
from collections import deque

from competition_olympics_env_wrapper import CompetitionOlympicsEnvWrapper
from olympics_env.chooseenv import make

GAME_NAME_LIST = [
    'running-competition', 'wrestling', 'football', 'table-hockey'
]


def print_all_competition_olympics_info(config):
    env = make(config.ENV_NAME, seed=42)          #build environment
    # print(env)

    from inspect import getmembers, ismethod

    member_list = [o for o in getmembers(env)]
    method_list = [o for o in getmembers(env, predicate=ismethod)]
    method_name_list = [o[0] for o in getmembers(env, predicate=ismethod)]

    idx = 0
    for method in method_list:
            idx += 1
            print(idx, method)

    idx = 0
    for member in member_list:
        if not member[0].startswith("__") and not member[0].startswith("_") and member[0] not in method_name_list:
            idx += 1
            print(idx, member)

    print("OBSERVATION TYPE:", type(env.current_state[0]['agent_obs']))
    print("OBSERVATION SHAPE:", env.current_state[0]['agent_obs'].shape)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')

    width = env.env_core.view_setting['width'] + 2 * env.env_core.view_setting['edge']
    height = env.env_core.view_setting['height'] + 2 * env.env_core.view_setting['edge']
    print(f'Game board width: {width}')
    print(f'Game board height: {height}')

    act_dim = env.action_dim
    obs_dim = (40, 40)
    print(f'action dimension: {act_dim}')
    print(f'observation dimension: {obs_dim}')


class Dummy_Agent:
    def __init__(self, gym_wrapper=False):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.gym_wrapper = gym_wrapper

    def seed(self, seed=None):
        random.seed(seed)

    def get_action(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        if self.gym_wrapper:
            return [force, angle]
        else:
            return [[force], [angle]]


def competition_olympics_controlled_agent_test(config, controlled_agent_index=1, render=False):
    controlled_agent = Dummy_Agent(gym_wrapper=False)
    opponent_agent = Dummy_Agent(gym_wrapper=False)

    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)
    record_draw = deque(maxlen=100)
    print(
        "CompetitionOlympicsEnv INITIALIZED with controlled_agent_index = {0}".format(controlled_agent_index),
        flush=True
    )

    env = make(config.ENV_NAME)  # build environment

    MAX_EPISODE = 100

    for episode in range(MAX_EPISODE):
        observation = env.reset()
        #print("RESET & NEW ENV - {0}".format(i + 1))
        done = False

        # if render:
        #     env.env_core.render()

        observation_opponent_agent = np.array(observation[1 - controlled_agent_index]['obs']['agent_obs'])
        observation_controlled_agent = observation[controlled_agent_index]['obs']['agent_obs']

        reward = None
        episode_reward = 0.0

        time_steps = 0

        while not done:
            time_steps += 1
            action_opponent = opponent_agent.get_action(observation_opponent_agent)
            action_controlled = controlled_agent.get_action(observation_controlled_agent)

            action = [action_opponent, action_controlled] if controlled_agent_index == 1 else [action_controlled, action_opponent]

            next_observation, reward, done, info_before, info_after = env.step(action)

            next_observation_opponent_agent = next_observation[1 - controlled_agent_index]['obs']['agent_obs']
            next_observation_controlled_agent = next_observation[controlled_agent_index]['obs']['agent_obs']

            reward_opponent = reward[1 - controlled_agent_index]
            reward_controlled = reward[controlled_agent_index]

            print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, "
                  "info_before: {5}, info_after: {6}".format(
                observation_controlled_agent.shape, action, next_observation_controlled_agent.shape, reward_controlled, done,
                info_after, info_after
            ))
            observation_controlled_agent = next_observation_controlled_agent
            observation_opponent_agent = next_observation_opponent_agent

            # if render:
            #     env.env_core.render()

        win_is = 1 if reward[controlled_agent_index] > reward[1 - controlled_agent_index] else 0
        win_is_op = 1 if reward[controlled_agent_index] < reward[1 - controlled_agent_index] else 0
        draw_is = 1 if reward[controlled_agent_index] == reward[1 - controlled_agent_index] else 0
        record_win.append(win_is)
        record_win_op.append(win_is_op)
        record_draw.append(draw_is)

        global_max_value_in_obs = 0.0

        print_episode_stat(
            episode, time_steps, controlled_agent_index, reward, episode_reward, global_max_value_in_obs,
            record_win, record_win_op, record_draw
        )


def competition_olympics_controlled_agent_test_2(config, controlled_agent_index=1, render=False):
    agent = Dummy_Agent(gym_wrapper=True)

    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)
    record_draw = deque(maxlen=100)
    print(
        "CompetitionOlympicsEnv INITIALIZED with controlled_agent_index = {0}".format(controlled_agent_index),
        flush=True
    )
    env = make(config.ENV_NAME)  # build environment
    env = CompetitionOlympicsEnvWrapper(env, controlled_agent_index=controlled_agent_index)

    MAX_EPISODE = 100
    global_max_value_in_obs = 0

    for episode in range(MAX_EPISODE):
        observation = env.reset()

        done = False

        reward = None
        info = None

        episode_reward = 0.0

        time_steps = 0

        while not done:
            time_steps += 1
            action = agent.get_action(observation)
            next_observation, reward, done, info = env.step(action)

            # print("Observation: {0}, Action: {1}, next_observation: {2}, Reward: {3}, Done: {4}, info: {5}".format(
            #     next_observation.shape, action, next_observation.shape, reward, done, info
            # ))

            episode_reward += reward
            observation = next_observation

            # print(observation, local_max_value_in_obs)
            local_max_value_in_obs = np.amax(observation)
            if local_max_value_in_obs > global_max_value_in_obs:
                global_max_value_in_obs = local_max_value_in_obs

        win_is = 1 if 'win_controlled_agent' in info and info['win_controlled_agent'] else 0
        win_is_op = 1 if 'win_opponent_agent' in info and info['win_opponent_agent'] else 0
        draw_is = 1 if 'draw' in info and info['draw'] else 0
        record_win.append(win_is)
        record_win_op.append(win_is_op)
        record_draw.append(draw_is)

        print_episode_stat(
            episode, time_steps, controlled_agent_index, reward, episode_reward, global_max_value_in_obs,
            record_win, record_win_op, record_draw
        )


def print_episode_stat(
        episode, time_steps, controlled_agent_index, reward, episode_reward, global_max_value_in_obs,
        record_win, record_win_op, record_draw
):
    print(
        "Episode: {0}, Elapsed Time Steps: {1}, Controlled agent: {2}, Last Reward: {3},"
        " Episode Reward: {4}, Max Value in Obs.: {5}".format(
            episode + 1, time_steps, controlled_agent_index, reward, episode_reward, global_max_value_in_obs
        ), end=""
    )

    if controlled_agent_index == 0:
        print(
            ", Controlled Win : Opponent Win : Draw = {0:.2f} : {1:.2f} : {2:.2f}".format(
                sum(record_win) / len(record_win),
                sum(record_win_op) / len(record_win_op),
                sum(record_draw) / len(record_draw)
            ), end="\n\n"
        )
    else:
        print(
            ", Opponent Win : Controlled Win : Draw = {0:.2f} : {1:.2f} : {2:.2f}".format(
                sum(record_win_op) / len(record_win_op),
                sum(record_win) / len(record_win),
                sum(record_draw) / len(record_draw)
            ), end="\n\n"
        )


if __name__ == '__main__':
    from link_rl.a_configuration.b_single_config.competition_olympics.config_competition_olympics_integrated import \
        ConfigCompetitionOlympicsIntegratedPpo

    config = ConfigCompetitionOlympicsIntegratedPpo()

    print_all_competition_olympics_info(config)

    #competition_olympics_controlled_agent_test(config, controlled_agent_index=1)

    #competition_olympics_controlled_agent_test_2(config, controlled_agent_index=1)
