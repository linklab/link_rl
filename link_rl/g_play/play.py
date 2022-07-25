import sys
import os
import warnings

import numpy as np

from link_rl.a_configuration.a_base_config.a_environments.dm_control import ConfigDmControl
from link_rl.a_configuration.a_base_config.a_environments.somo_gym import ConfigSomoGym
from link_rl.a_configuration.a_base_config.a_environments.unity.config_unity_box import ConfigUnityGymEnv
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent2DConvolutionalModel, ConfigRecurrent1DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.f_main.supports.tester import Tester
from link_rl.h_utils.commons import set_config, get_specific_env_name
from link_rl.h_utils.commons_rl import get_agent

warnings.filterwarnings("ignore")

from link_rl.h_utils.types import AgentMode, AgentType

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from link_rl.f_main.config_single import config
from link_rl.h_utils.commons import model_load, get_single_env, get_env_info

is_recurrent_model = any([
    isinstance(config.MODEL_PARAMETER, ConfigRecurrentLinearModel),
    isinstance(config.MODEL_PARAMETER, ConfigRecurrent1DConvolutionalModel),
    isinstance(config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
])

dm_control_episode_steps = 0
dm_control_episode_reward = 0.0


def dm_control_play(env, agent, n_episodes):
    global dm_control_episode_steps, dm_control_episode_reward
    from dm_control import viewer

    def get_action(time_step):
        global dm_control_episode_steps, dm_control_episode_reward
        dm_control_episode_steps += 1
        observation = env.get_observation(time_step, dm_control_episode_steps == 0)
        if config.AGENT_TYPE not in [AgentType.TDMPC]:
            observation = np.expand_dims(observation, axis=0)
        actions = agent.get_action(observation, mode=AgentMode.PLAY)
        actions_np = np.asarray(actions)
        env.action_repeat_for_play(actions)
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

    agent = get_agent(observation_space, action_space, config, need_train=False)

    env_name = get_specific_env_name(config=config)

    model_load(
        agent=agent, env_name=env_name, agent_type_name=config.AGENT_TYPE.name, config=config
    )

    if isinstance(config, ConfigDmControl):
        test_env = get_single_env(config, config.NO_TEST_GRAPHICS, train_mode=False)
        dm_control_play(test_env, agent, n_episodes=1)
        test_env.close()
    else:
        if isinstance(config, ConfigSomoGym):
            player = Tester(agent=agent, config=config, play=True, max_episode_step=1_000)
        else:
            player = Tester(agent=agent, config=config, play=True)

        player.play_for_testing(n_episodes=n_episodes)

        player.test_env.close()


if __name__ == "__main__":
    N_EPISODES = 5
    main_play(n_episodes=N_EPISODES)
