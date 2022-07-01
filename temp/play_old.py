import sys
import time
import os
import warnings

import numpy as np

from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_mujoco import ConfigMujoco
from link_rl.a_configuration.a_base_config.a_environments.open_ai_gym.config_gym_atari import ConfigGymAtari
from link_rl.a_configuration.a_base_config.a_environments.pybullet.config_gym_pybullet import ConfigBullet
from link_rl.a_configuration.a_base_config.a_environments.unity.config_unity_box import ConfigUnityGymEnv
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent2DConvolutionalModel, ConfigRecurrent1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.g_utils.commons import set_config, get_specific_env_name
from link_rl.g_utils.commons_rl import get_agent

warnings.filterwarnings("ignore")

from gym.spaces import Discrete, Box

from link_rl.g_utils.types import AgentMode

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from link_rl.e_main.config_single import config
from link_rl.g_utils.commons import model_load, get_single_env, get_env_info

is_recurrent_model = any([
    isinstance(config.MODEL_PARAMETER, ConfigRecurrentLinearModel),
    isinstance(config.MODEL_PARAMETER, ConfigRecurrent1DConvolutionalModel),
    isinstance(config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
])

dm_control_episode_steps = 0
dm_control_episode_reward = 0.0


def play(env, agent, n_episodes):
    for i in range(n_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        if isinstance(config, ConfigGymAtari):
            observation = env.reset()
        elif isinstance(config, (ConfigMujoco, ConfigBullet)):
            env.render()
            observation = env.reset()
        else:
            observation = env.reset()
            env.render()

        observation = np.expand_dims(observation, axis=0)
        if is_recurrent_model:
            agent.model.init_recurrent_hidden()
            observation = [(observation, agent.model.recurrent_hidden)]

        episode_steps = 0

        while True:
            episode_steps += 1
            action = agent.get_action(observation, mode=AgentMode.PLAY)

            if isinstance(agent.action_space, Discrete):
                if action.ndim == 0:
                    scaled_action = action
                elif action.ndim == 1:
                    scaled_action = action[0]
                else:
                    raise ValueError()
            elif isinstance(agent.action_space, Box):
                if action.ndim == 1:
                    if agent.action_scale is not None:
                        scaled_action = action * agent.action_scale + agent.action_bias
                    else:
                        scaled_action = action
                elif action.ndim == 2:
                    if agent.action_scale is not None:
                        scaled_action = action[0] * agent.action_scale + agent.action_bias
                    else:
                        scaled_action = action[0]
                else:
                    raise ValueError()
            else:
                raise ValueError()

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(scaled_action)
            next_observation = np.expand_dims(next_observation, axis=0)
            if is_recurrent_model:
                next_observation = [(next_observation, agent.model.recurrent_hidden)]

            if not isinstance(config, ConfigGymAtari):
                env.render()

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            time.sleep(0.01)
            if done:
                break

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i + 1, episode_steps, episode_reward
        ))


def dm_control_play(env, agent, n_episodes):
    global dm_control_episode_steps, dm_control_episode_reward
    from dm_control import viewer

    def get_action(time_step):
        global dm_control_episode_steps, dm_control_episode_reward
        dm_control_episode_steps += 1
        observation = env.get_observation(time_step)
        actions = agent.get_action(observation, mode=AgentMode.PLAY)
        actions_np = np.asarray(actions)
        dm_control_episode_reward += time_step.reward or 0
        return actions_np

    for i in range(n_episodes):
        env.reset()
        viewer.launch(env.original_env, policy=get_action)
        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i + 1, dm_control_episode_steps, dm_control_episode_reward
        ))
        dm_control_episode_steps = 0
        dm_control_episode_reward = 0.0


def main_play(n_episodes):
    set_config(config)

    observation_space, action_space = get_env_info(config)

    if isinstance(config, ConfigUnityGymEnv):
        config.NO_TEST_GRAPHICS = False

    env = get_single_env(config, config.NO_TEST_GRAPHICS, train_mode=False)

    agent = get_agent(observation_space, action_space, config, need_train=False)

    env_name = get_specific_env_name(config=config)

    model_load(
        agent=agent, env_name=env_name, agent_type_name=config.AGENT_TYPE.name, config=config
    )

    agent.model.eval()

    if isinstance(config, ConfigDmControl):
        dm_control_play(env, agent, n_episodes=1)
    else:
        play(env, agent, n_episodes=n_episodes)

    env.close()


if __name__ == "__main__":
    N_EPISODES = 5
    main_play(n_episodes=N_EPISODES)