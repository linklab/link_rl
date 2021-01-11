# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
import torch
import os, sys
import numpy as np

from codes.e_utils.actions import EpsilonGreedySomeTimesBlowDQNActionSelector, \
    EpsilonGreedySomeTimesBlowDDPGActionSelector, ArgmaxActionSelector, EpsilonGreedyDDPGActionSelector, \
    ContinuousNormalActionSelector

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.e_utils import rl_utils
from codes.e_utils.common_utils import load_model
from codes.e_utils.logger import get_logger
from codes.e_utils.names import RLAlgorithmName, EnvironmentName

MODEL_ZOO_SAVE_DIR = os.path.join(PROJECT_HOME, "codes", "g_play", "model_zoo")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

my_logger = get_logger("openai_pendulum_ddpg")


def play_main(params):
    env = rl_utils.get_environment(owner="actual_worker", params=params)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    agent, epsilon_tracker = rl_utils.get_rl_agent(env=env, worker_id=0, params=params, device=device)

    load_model(MODEL_ZOO_SAVE_DIR, params.ENVIRONMENT_ID.value, agent)

    if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0]:
        if params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
            action_selector = EpsilonGreedySomeTimesBlowDQNActionSelector(
                epsilon=0.0
            )
        elif params.RL_ALGORITHM == RLAlgorithmName.DDPG_FAST_V0:
            action_selector = EpsilonGreedySomeTimesBlowDDPGActionSelector(
                epsilon=0.0, ou_enabled=False, scale_factor=params.ACTION_SCALE
            )
        else:
            raise ValueError()
    else:
        if params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_PPO_FAST_V0:
            action_selector = ContinuousNormalActionSelector()
        elif params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
            action_selector = ArgmaxActionSelector()
        elif params.RL_ALGORITHM == RLAlgorithmName.DDPG_FAST_V0:
            action_selector = EpsilonGreedyDDPGActionSelector(
                epsilon=0.0, ou_enabled=False, scale_factor=params.ACTION_SCALE
            )
        else:
            raise ValueError()

    agent.action_selector = action_selector

    num_step = 0
    num_episode = 0

    while True:
        done = False
        episode_reward = 0

        state = env.reset()

        num_episode += 1
        num_episode_step = 0
        while not done:
            num_step += 1
            num_episode_step += 1

            state = np.expand_dims(state, axis=0)

            action, _, = agent(state)

            next_state, reward, done, info = env.step(action[0])
            state = next_state
            episode_reward += reward

            if num_step % 1000 == 0:
                print("EPISODE: {0}, EPISODE STEPS: {1}, TOTAL STEPS: {2}".format(
                    num_episode, num_episode_step, num_step
                ))

        print("EPISODE: {0}, EPISODE STEPS: {1}, TOTAL STEPS: {2}, EPISODE DONE --> EPISODE REWARD: {3}".format(
            num_episode, num_episode_step, num_step, episode_reward
        ))

        time.sleep(0.1)


if __name__ == "__main__":
    from codes.a_config.parameters import PARAMETERS as parameters
    params = parameters
    play_main(params)
