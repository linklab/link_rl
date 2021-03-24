# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
import torch
import os, sys
import numpy as np

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.e_utils.actions import EpsilonGreedySomeTimesBlowDQNActionSelector, \
    SomeTimesBlowDDPGActionSelector, ArgmaxActionSelector, DDPGActionSelector, \
    ContinuousNormalActionSelector, DiscreteCategoricalActionSelector
from codes.e_utils.rl_utils import get_environment_input_output_info, MODEL_ZOO_SAVE_DIR, MODEL_SAVE_FILE_PREFIX
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import load_model
from codes.e_utils.logger import get_logger
from codes.e_utils.names import RLAlgorithmName, EnvironmentName, AgentMode
from codes.e_utils.rl_utils import get_environment_input_output_info


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_logger = get_logger("openai_pendulum_ddpg")


def play_main(params, env):
    input_shape, action_shape, num_outputs, action_min, action_max = get_environment_input_output_info(env)
    agent = rl_utils.get_rl_agent(
        input_shape, action_shape, num_outputs, action_min, action_max, worker_id=-1, params=params, device=device
    )
    load_model(MODEL_ZOO_SAVE_DIR, MODEL_SAVE_FILE_PREFIX, agent, inquery=False)
    agent.agent_mode = AgentMode.PLAY

    num_step = 0
    num_episode = 0

    while True:
        done = False
        episode_reward = 0

        if params.ENVIRONMENT_ID in [
            EnvironmentName.PYBULLET_ANT_V0, EnvironmentName.PYBULLET_HALF_CHEETAH_V0,
            EnvironmentName.PYBULLET_INVERTED_DOUBLE_PENDULUM_V0
        ]:
            env.render()
        state = env.reset()

        num_episode += 1
        num_episode_step = 0
        while not done:
            env.render()

            num_step += 1
            num_episode_step += 1

            state = np.expand_dims(state, axis=0)

            action, _, = agent(state)

            if params.ACTION_SCALE:
                action = params.ACTION_SCALE * action[0]
            else:
                action = action[0]

            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward

            # if num_step % 1000 == 0:
            #     print("EPISODE: {0}, EPISODE STEPS: {1}, TOTAL STEPS: {2}".format(
            #         num_episode, num_episode_step, num_step
            #     ))

            if params.ENVIRONMENT_ID not in [
                EnvironmentName.PENDULUM_MATLAB_V0,
                EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0,
                EnvironmentName.REAL_DEVICE_RIP,
                EnvironmentName.REAL_DEVICE_DOUBLE_RIP
            ]:
                time.sleep(0.01)

        print("EPISODE: {0}, EPISODE STEPS: {1}, TOTAL STEPS: {2}, EPISODE DONE --> EPISODE REWARD: {3}".format(
            num_episode, num_episode_step, num_step, episode_reward
        ))

        time.sleep(0.1)


if __name__ == "__main__":
    from codes.a_config.parameters import PARAMETERS as parameters
    params = parameters

    env = rl_utils.get_single_environment(params=params, mode=AgentMode.PLAY)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    play_main(params, env)
