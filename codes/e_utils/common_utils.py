import glob
import os
import random
from typing import Optional
import numpy as np
import math

from gym.spaces import Box
from matplotlib import pyplot as plt
import gym
import torch

from codes.b_environments.or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv
from codes.b_environments.or_gym.envs.classic_or.tsp import TSPEnv, TSPDistCost

from codes.d_agents.a0_base_agent import float32_preprocessor

#https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f
from codes.e_utils import wrappers
from codes.e_utils.names import AgentMode


def stretched_exponential_decay(epsilon_start, epsilon_minimum, epsilon_end_step, current_step):
    end_step = epsilon_end_step
    if current_step < end_step:
        A = 0.5
        B = 0.3
        C = 0.1
        standardized_time = (current_step - A * end_step) / (B * end_step)
        cosh = np.cosh(math.exp(-standardized_time))
        epsilon = epsilon_start - (1 / cosh + (current_step * C / end_step))
        if epsilon >= epsilon_start:
            epsilon = epsilon_start
        elif epsilon <= epsilon_minimum:
            epsilon = epsilon_minimum
    else:
        epsilon = epsilon_minimum
    return epsilon


def print_params(params_class):
    print('\n' + '################ Parameters ################')
    for param in dir(params_class):
        if not param.startswith("__"):
            print("{0}: {1}".format(param, getattr(params_class, param)))
    print('############################################')
    print()


def print_fast_rl_params(params_class):
    print('\n' + '################ Parameters ################')
    for param in dir(params_class):
        if not param.startswith("__") and not param.startswith("MQTT"):
            print("{0}: {1}".format(param, getattr(params_class, param)))
    print('############################################')
    print()


def set_global_seeds(seed):
    myseed = seed + 1000 if seed is not None else None
    try:
        torch.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


def make_atari_env(env_id, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    """
    set_global_seeds(seed)

    env = gym.make(env_id)
    env = wrappers.wrap_dqn(env)
    return env


def make_gym_env(env_id, rank=0, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    set_global_seeds(seed)

    env = gym.make(env_id)
    env.seed(seed + rank)

    return env


def make_or_gym_env(env_name, env_config, rank=0, seed=0):
    set_global_seeds(seed)
    if env_name == 'Knapsack-v2':
        env = BoundedKnapsackEnv(env_config=env_config)
    elif env_name == "TSP-v0":
        env = TSPEnv(env_config=env_config)
    elif env_name == "TSP-v1":
        env = TSPDistCost(env_config=env_config)
    else:
        raise ValueError(env_name)

    env.set_seed(seed + rank)

    return env


# https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f
def epsilon_scheduled(current_episode, max_episodes, initial_epsilon, final_epsilon):
    A = 0.05     # spend more time, either on Exploration or on Exploitation
    B = 0.04     # the slope of transition region between Exploration to Exploitation zone
    C = 0.1     # the steepness of left and right tail of the graph
    standardized_time = (current_episode - A * max_episodes)/(B * max_episodes)
    cosh = np.cosh(math.exp(-standardized_time))
    epsilon = max(initial_epsilon - (1 / cosh + (current_episode * C / max_episodes)), final_epsilon)
    return epsilon


def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1.0 - alpha) * val


def unpack_batch_for_a2c(batch, net, params, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, target values variable
    """
    states, actions, rewards, not_done_idx, last_states = [], [], [], [], []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = float32_preprocessor(states).to(device)
    actions_v = float32_preprocessor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)

    if not_done_idx:
        last_states_v = float32_preprocessor(last_states).to(device)
        last_values_v = net.base.forward_critic(last_states_v)
        last_values_np = last_values_v.data.cpu().numpy()[:, 0] * params.GAMMA ** params.N_STEP
        rewards_np[not_done_idx] += last_values_np

    target_action_values_v = float32_preprocessor(rewards_np).to(device)

    return states_v, actions_v, target_action_values_v


def remove_models(model_save_dir, env_name, agent):
    files = glob.glob(os.path.join(
        model_save_dir, "{0}_{1}_{2}_*.pth".format(env_name, agent.__name__, agent.model.__name__))
    )
    for f in files:
        os.remove(f)


def save_model(model_save_dir, env_name, agent, step, episode_reward):
    model_save_filename = os.path.join(
        model_save_dir, "{0}_{1}_{2}_{3}_{4:.2f}.pth".format(
            env_name, agent.__name__, agent.model.__name__, step, float(episode_reward)
        )
    )
    torch.save(agent.model.state_dict(), model_save_filename)
    return model_save_filename


def load_model(model_save_dir, env_name, agent, step=None):
    if step:
        saved_models = glob.glob(os.path.join(
            model_save_dir, "{0}_{1}_{2}_{3}_*.pth".format(env_name, agent.__name__, agent.model.__name__, step)
        ))

    else:
        saved_models = glob.glob(os.path.join(
            model_save_dir, "{0}_{1}_{2}_*.pth".format(env_name, agent.__name__, agent.model.__name__)
        ))

    saved_models.sort(key=lambda filename: int(filename.split("/")[-1].split("_")[-2]))
    assert len(saved_models) > 0, "※※※※※※※※※※ There is no model !!!: {0} ※※※※※※※※※※".format(saved_models)

    saved_model = saved_models[-1]
    print("SAVED MODEL FILE NAME: {0}".format(saved_model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params = torch.load(saved_model, map_location=device)

    agent.model.load_state_dict(model_params)


def agent_model_test(params, test_env, agent):
    agent.agent_mode = AgentMode.TEST

    num_step = 0

    episode_rewards = np.zeros(params.TEST_NUM_EPISODES)

    for test_episode in range(params.TEST_NUM_EPISODES):
        done = False
        episode_reward = 0

        state = test_env.reset()

        num_episode_step = 0
        while not done:
            num_step += 1
            num_episode_step += 1

            state = np.expand_dims(state, axis=0)

            action, _, = agent(state)

            next_state, reward, done, info = test_env.step(action[0])
            state = next_state
            episode_reward += reward

        episode_rewards[test_episode] = episode_reward

    return np.mean(episode_rewards), np.std(episode_rewards)



# def print_environment_info(env, params):
#     print(f"env id: {params.ENVIRONMENT_ID}")
#     print(f"number of unique envs: {params.NUM_ENVIRONMENTS}")
#     print(f"env.observation_space: {env[0].observation_space}")
#     if isinstance(env[0].observation_space, Box):
#         print(f"observation low: {[min_value for min_value in env[0].observation_space.low]}")
#         print(f"observation high: {[max_value for max_value in env[0].observation_space.high]}")
#
#     print(f"env.action_space: {env[0].action_space}")
#     if isinstance(env[0].action_space, Box):
#         print(f"action low: {[min_value for min_value in env[0].action_space.low]}")
#         print(f"action high: {[max_value for max_value in env[0].action_space.high]}")


def print_environment_info(env, params):
    print(f"env id: {params.ENVIRONMENT_ID}")
    print(f"number of unique envs: {params.NUM_ENVIRONMENTS}")
    print(f"env.single_observation_space: {env.single_observation_space}")
    if isinstance(env.single_observation_space, Box):
        print(f"single_observation low: {[min_value for min_value in env.single_observation_space.low]}")
        print(f"single_observation high: {[max_value for max_value in env.single_observation_space.high]}")

    print(f"env.single_action_space: {env.single_action_space}")
    if isinstance(env.single_action_space, Box):
        print(f"single_action low: {[min_value for min_value in env.single_action_space.low]}")
        print(f"single_action high: {[max_value for max_value in env.single_action_space.high]}")


def print_agent_info(agent, epsilon_tracker, params):
    print(f"Model: {params.DEEP_LEARNING_MODEL}")
    print(f"Algorithm: {params.RL_ALGORITHM}")
    print(f"Train Action Selector: {agent.train_action_selector}")
    print(f"Test and Play Action Selector: {agent.test_and_play_action_selector}")
    print(f"Epsilon Tracker: {epsilon_tracker if epsilon_tracker else None}")
    print(f"Optimizer: {params.OPTIMIZER}")


if __name__=="__main__":
    # max_episode = 20000000
    max_episode = 200
    initial_epsilon = 1.0
    final_epsilon = 0.01
    epsilon_list = []
    plt.plot(
        [x for x in range(max_episode)],
        [epsilon_scheduled(x, max_episode, initial_epsilon, final_epsilon) for x in range(max_episode)]
    )
    plt.show()