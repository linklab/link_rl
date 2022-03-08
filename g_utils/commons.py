import collections
import datetime
import numpy as np
from torch import nn
import torch.nn.functional as F

import gym
import torch
import os
import torch.multiprocessing as mp
import wandb
from gym.spaces import Discrete, Box
from gym.vector import AsyncVectorEnv
import plotly.graph_objects as go

from a_configuration.a_base_config.config_parse import SYSTEM_USER_NAME
from a_configuration.a_base_config.a_environments.unity.config_unity_box import ConfigUnityGymEnv
from a_configuration.a_base_config.c_models.config_convolutional_models import ConfigConvolutionalModel
from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
from a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
from a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from b_environments.wrapper import MakeBoxFrozenLake
from g_utils.types import AgentType, ActorCriticAgentTypes, ModelType, LayerActivationType, LossFunctionType, \
    OffPolicyAgentTypes


def model_save(model, env_name, agent_type_name, test_episode_reward_avg, test_episode_reward_std, config):
    env_model_home = os.path.join(config.MODEL_SAVE_DIR, env_name)
    if not os.path.exists(env_model_home):
        os.mkdir(env_model_home)

    agent_model_home = os.path.join(config.MODEL_SAVE_DIR, env_name, agent_type_name)
    if not os.path.exists(agent_model_home):
        os.mkdir(agent_model_home)

    now = datetime.datetime.now()
    local_now = now.astimezone()
    file_name = "{0:4.1f}_{1:3.1f}_{2}_{3}_{4}_{5}_{6}.pth".format(
        test_episode_reward_avg, test_episode_reward_std, local_now.year, local_now.month, local_now.day,
        env_name, agent_type_name
    )

    torch.save(model.state_dict(), os.path.join(agent_model_home, file_name))


def model_load(model, env_name, agent_type_name, file_name, config):
    agent_model_home = os.path.join(config.MODEL_SAVE_DIR, env_name, agent_type_name)
    model_params = torch.load(os.path.join(agent_model_home, file_name), map_location=torch.device('cpu'))
    model.load_state_dict(model_params)


def set_config(config):
    if config.MODEL_TYPE in (
            ModelType.TINY_LINEAR, ModelType.SMALL_LINEAR, ModelType.SMALL_LINEAR_2,
            ModelType.MEDIUM_LINEAR, ModelType.LARGE_LINEAR
    ):
        from a_configuration.a_base_config.c_models.config_linear_models import ConfigLinearModel
        config.MODEL_PARAMETER = ConfigLinearModel(config.MODEL_TYPE)

    elif config.MODEL_TYPE in (
            ModelType.TINY_CONVOLUTIONAL, ModelType.SMALL_CONVOLUTIONAL,
            ModelType.MEDIUM_CONVOLUTIONAL, ModelType.LARGE_CONVOLUTIONAL
    ):
        from a_configuration.a_base_config.c_models.config_convolutional_models import ConfigConvolutionalModel
        config.MODEL_PARAMETER = ConfigConvolutionalModel(config.MODEL_TYPE)

    elif config.MODEL_TYPE in (
            ModelType.SMALL_RECURRENT, ModelType.MEDIUM_RECURRENT, ModelType.LARGE_RECURRENT
    ):
        from a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
        config.MODEL_PARAMETER = ConfigRecurrentLinearModel(config.MODEL_TYPE)

    elif config.MODEL_TYPE in (
            ModelType.SMALL_RECURRENT_CONVOLUTIONAL, ModelType.MEDIUM_RECURRENT_CONVOLUTIONAL,
            ModelType.LARGE_RECURRENT_CONVOLUTIONAL
    ):
        from a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
        config.MODEL_PARAMETER = ConfigRecurrentConvolutionalModel(config.MODEL_TYPE)

    else:
        raise ValueError()

    if config.LAYER_ACTIVATION_TYPE == LayerActivationType.LEAKY_RELU:
        config.LAYER_ACTIVATION = nn.LeakyReLU
    elif config.LAYER_ACTIVATION_TYPE == LayerActivationType.ELU:
        config.LAYER_ACTIVATION = nn.ELU
    elif config.LAYER_ACTIVATION_TYPE == LayerActivationType.PReLU:
        config.LAYER_ACTIVATION = nn.PReLU
    elif config.LAYER_ACTIVATION_TYPE == LayerActivationType.SELU:
        config.LAYER_ACTIVATION = nn.SELU
    else:
        raise ValueError()

    if config.VALUE_NETWORK_LAYER_ACTIVATION_TYPE == LayerActivationType.LEAKY_RELU:
        config.VALUE_NETWORK_LAYER_ACTIVATION = nn.LeakyReLU
    elif config.VALUE_NETWORK_LAYER_ACTIVATION_TYPE == LayerActivationType.ELU:
        config.VALUE_NETWORK_LAYER_ACTIVATION = nn.ELU
    elif config.VALUE_NETWORK_LAYER_ACTIVATION_TYPE == LayerActivationType.PReLU:
        config.VALUE_NETWORK_LAYER_ACTIVATION = nn.PReLU
    elif config.VALUE_NETWORK_LAYER_ACTIVATION_TYPE == LayerActivationType.SELU:
        config.VALUE_NETWORK_LAYER_ACTIVATION = nn.SELU
    else:
        raise ValueError()

    if config.LOSS_FUNCTION_TYPE == LossFunctionType.MSE_LOSS:
        config.LOSS_FUNCTION = F.mse_loss
    elif config.LOSS_FUNCTION_TYPE == LossFunctionType.HUBER_LOSS:
        config.LOSS_FUNCTION = F.huber_loss
    else:
        raise ValueError()

    if config.AGENT_TYPE in OffPolicyAgentTypes:
        config.MIN_BUFFER_SIZE_FOR_TRAIN = config.BATCH_SIZE * 5

    elif config.AGENT_TYPE == AgentType.REINFORCE:
        config.BUFFER_CAPACITY = -1
        config.N_STEP = 1

    elif config.AGENT_TYPE == AgentType.A2C:
        config.BUFFER_CAPACITY = config.BATCH_SIZE
        config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 30

    elif config.AGENT_TYPE == AgentType.A3C:
        config.BUFFER_CAPACITY = config.BATCH_SIZE
        config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS * config.N_ACTORS / 2
        assert config.N_ACTORS > 1

    elif config.AGENT_TYPE == AgentType.PPO:
        config.BUFFER_CAPACITY = config.BATCH_SIZE
        config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 10 * config.PPO_K_EPOCH

    elif config.AGENT_TYPE == AgentType.PPO_TRAJECTORY:
        config.PPO_TRAJECTORY_SIZE = config.BATCH_SIZE * 10
        config.BUFFER_CAPACITY = config.PPO_TRAJECTORY_SIZE
        config.CONSOLE_LOG_INTERVAL_TRAINING_STEPS = 10 * config.PPO_K_EPOCH

    elif config.AGENT_TYPE == AgentType.MUZERO:
        pass

    else:
        raise ValueError()


def print_base_info(config):
    n_cpu_cores = mp.cpu_count()
    print("{0:55} {1:55}".format(
        "DEVICE: {0}".format(config.DEVICE),
        "CPU CORES: {0}".format(n_cpu_cores),
    ), end="\n")

    print("{0:55} {1:55} {2:55}".format(
        "N_ACTORS: {0}".format(config.N_ACTORS),
        "ENVS PER ACTOR: {0}".format(config.N_VECTORIZED_ENVS),
        "TOTAL NUMBERS OF ENVS: {0}".format(
            config.N_ACTORS * config.N_VECTORIZED_ENVS
        )
    ))

    print("PROJECT_HOME: {0}".format(config.PROJECT_HOME))

    if hasattr(config, "MODEL_SAVE_DIR"):
        print("MODEL_SAVE_DIR: {0}".format(config.MODEL_SAVE_DIR))

    if hasattr(config, "COMPARISON_RESULTS_SAVE_DIR"):
        print("COMPARISON_RESULTS_SAVE_DIR: {0}".format(config.COMPARISON_RESULTS_SAVE_DIR))

    print("UNITY_ENV_DIR: {0}".format(config.UNITY_ENV_DIR))


def print_basic_info(observation_space=None, action_space=None, config=None):
    print('\n' + '#' * 81 + " Base Configs " + '#' * 82)

    print_base_info(config)

    print('-' * 75 + " Config " + '-' * 75)

    items = []

    for param in dir(config):
        if not param.startswith("__") and param not in [
            "MODEL_PARAMETER", "NEURONS_PER_FULLY_CONNECTED_LAYER", "OUT_CHANNELS_PER_LAYER", "KERNEL_SIZE_PER_LAYER",
            "STRIDE_PER_LAYER", "EPISODE_REWARD_AVG_SOLVED", "EPISODE_REWARD_STD_SOLVED", "UNITY_ENV_DIR",
            "MODEL_SAVE_DIR", "PROJECT_HOME", "LAYER_ACTIVATION", "LAYER_ACTIVATION_TYPE",
            "VALUE_NETWORK_LAYER_ACTIVATION", "VALUE_NETWORK_LAYER_ACTIVATION_TYPE",
            "LOSS_FUNCTION", "ENV_NAME", "PLAY_MODEL_FILE_NAME", "MODEL_TYPE", "LEARNING_RATE", "ACTOR_LEARNING_RATE",
            "ALPHA_LEARNING_RATE"
        ]:
            if param in [
                "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TRAINING_STEPS", "MAX_TRAINING_STEPS",
                "MIN_BUFFER_SIZE_FOR_TRAIN", "N_EPISODES_FOR_MEAN_CALCULATION",
                "TEST_INTERVAL_TRAINING_STEPS"
            ]:
                item = "{0}: {1:,}".format(param, getattr(config, param))
            else:
                item = "{0}: {1:}".format(param, getattr(config, param))
            items.append(item)

        if len(items) == 3:
            print("{0:55} {1:55} {2:55}".format(items[0], items[1], items[2]), end="\n")
            items.clear()

    if len(items) > 0:
        if len(items) == 2:
            print("{0:55} {1:55}".format(items[0], items[1]), end="\n")
            items.clear()
        else:
            print("{0:55}".format(items[0]), end="\n")
            items.clear()

    print_learning_rate_info(config)
    print_model_info(config)

    if observation_space and action_space:
        if observation_space and action_space:
            print('-' * 77 + " ENV " + '-' * 77)
        print_env_info(observation_space, action_space, config)

    print('#' * 182)
    print()


def print_comparison_basic_info(observation_space, action_space, config_c):
    print('\n' + '#' * 81 + " Base Config " + '#' * 82)

    print_base_info(config_c)
    print('-' * 71 + " Common Config " + '-' * 71)

    items = []

    for param in dir(config_c):
        if param == "AGENT_LABELS":
            item1 = "{0}: {1:}".format("N_AGENTS", len(getattr(config_c, param)))
            item2 = "{0}: {1:}".format(param, getattr(config_c, param))
            print("{0:55} {1:55}".format(item1, item2))
            continue

        if param == "COMPARISON_RESULTS_SAVE_DIR":
            item = "{0}: {1:}".format(param, getattr(config_c, param))
            print("{0:55}".format(item))
            continue

        if param == "AGENT_PARAMETERS":
            continue

        if not param.startswith("__"):
            if param in [
                "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TRAINING_STEPS", "MAX_TRAINING_STEPS",
                "MIN_BUFFER_SIZE_FOR_TRAIN", "N_EPISODES_FOR_MEAN_CALCULATION",
                "TEST_INTERVAL_TRAINING_STEPS"
            ]:
                item = "{0}: {1:,}".format(param, getattr(config_c, param))
            else:
                item = "{0}: {1:}".format(param, getattr(config_c, param))
            items.append(item)

        if len(items) == 3:
            print("{0:55} {1:55} {2:55}".format(items[0], items[1], items[2]), end="\n")
            items.clear()

    if len(items) > 0:
        if len(items) == 2:
            print("{0:55} {1:55}".format(items[0], items[1]), end="\n")
            items.clear()
        else:
            print("{0:55}".format(items[0]), end="\n")
            items.clear()

    for agent_idx, agent_config in enumerate(config_c.AGENT_PARAMETERS):
        print('-' * 76 + " Agent {0} ".format(agent_idx) + '-' * 76)
        for param in dir(agent_config):
            if not param.startswith("__") and param not in [
                "MODEL_PARAMETER", "NEURONS_PER_FULLY_CONNECTED_LAYER", "OUT_CHANNELS_PER_LAYER", "KERNEL_SIZE_PER_LAYER",
                "STRIDE_PER_LAYER", "EPISODE_REWARD_AVG_SOLVED", "EPISODE_REWARD_STD_SOLVED", "UNITY_ENV_DIR",
                "COMPARISON_RESULTS_SAVE_DIR", "PROJECT_HOME", "LAYER_ACTIVATION", "LAYER_ACTIVATION_TYPE",
                "VALUE_NETWORK_LAYER_ACTIVATION", "VALUE_NETWORK_LAYER_ACTIVATION_TYPE",
                "LOSS_FUNCTION", "ENV_NAME", "PLAY_MODEL_FILE_NAME", "MODEL_TYPE", "LEARNING_RATE", "ACTOR_LEARNING_RATE",
                "ALPHA_LEARNING_RATE"
            ]:
                if param in [
                    "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TRAINING_STEPS", "MAX_TRAINING_STEPS",
                    "MIN_BUFFER_SIZE_FOR_TRAIN", "N_EPISODES_FOR_MEAN_CALCULATION",
                    "TEST_INTERVAL_TRAINING_STEPS"
                ]:
                    item = "{0}: {1:,}".format(param, getattr(agent_config, param))
                else:
                    item = "{0}: {1:}".format(param, getattr(agent_config, param))
                items.append(item)

            if len(items) == 3:
                print("{0:55} {1:55} {2:55}".format(items[0], items[1], items[2]), end="\n")
                items.clear()

        if len(items) > 0:
            if len(items) == 2:
                print("{0:55} {1:55}".format(items[0], items[1]), end="\n")
                items.clear()
            else:
                print("{0:55}".format(items[0]), end="\n")
                items.clear()

        print_learning_rate_info(agent_config)

        print_model_info(agent_config)

    if observation_space and action_space:
        if observation_space and action_space:
            print('-' * 77 + " ENV " + '-' * 77)
        print_env_info(observation_space, action_space, config_c)

    print('#' * 182)
    print()


def print_learning_rate_info(config):
    print('-' * 72 + " LEARNING_RATE " + '-' * 72)

    ptr_str = "{:55} ".format("{0}: {1:}".format("LEARNING_RATE", config.LEARNING_RATE))

    if hasattr(config, "ACTOR_LEARNING_RATE"):
        ptr_str += "{:55} ".format("{0}: {1:}".format("ACTOR_LEARNING_RATE", config.ACTOR_LEARNING_RATE))

    if hasattr(config, "ALPHA_LEARNING_RATE"):
        ptr_str += "{:55}".format("{0}: {1:}".format("ALPHA_LEARNING_RATE", config.ALPHA_LEARNING_RATE))

    print(ptr_str, end="\n")


def print_model_info(config):
    if config.MODEL_PARAMETER is None:
        set_config(config)

    model_config = config.MODEL_PARAMETER
    print('-' * 74 + " MODEL_TYPE " + '-' * 74)

    if isinstance(model_config, ConfigLinearModel):
        item1 = "{0}: {1:}".format("MODEL_TYPE", config.MODEL_TYPE)
        item2 = "{0}: {1:}".format("NEURONS_PER_REPRESENTATION_LAYER", model_config.NEURONS_PER_REPRESENTATION_LAYER)
        item3 = "{0}: {1:}".format("NEURONS_PER_FULLY_CONNECTED_LAYER", model_config.NEURONS_PER_FULLY_CONNECTED_LAYER)
        print("{0:55} {1:55} {2:55}".format(item1, item2, item3), end="\n")
    elif isinstance(model_config, ConfigConvolutionalModel):
        item1 = "{0}: {1:}".format("MODEL_TYPE", config.MODEL_TYPE)
        item2 = "{0}: {1:}".format("OUT_CHANNELS_PER_LAYER", model_config.OUT_CHANNELS_PER_LAYER)
        item3 = "{0}: {1:}".format("KERNEL_SIZE_PER_LAYER", model_config.KERNEL_SIZE_PER_LAYER)
        print("{0:55} {1:55} {2:55}".format(item1, item2, item3, end="\n"))
        item1 = "{0}: {1:}".format("STRIDE_PER_LAYER", model_config.STRIDE_PER_LAYER)
        item2 = "{0}: {1:}".format("PADDING", model_config.PADDING)
        print("{0:55} {1:55}".format(item1, item2, end="\n"))
        item1 = "{0}: {1:}".format("NEURONS_PER_REPRESENTATION_LAYER", model_config.NEURONS_PER_REPRESENTATION_LAYER)
        item2 = "{0}: {1:}".format("NEURONS_PER_FULLY_CONNECTED_LAYER", model_config.NEURONS_PER_FULLY_CONNECTED_LAYER)
        print("{0:55} {1:55}".format(item1, item2), end="\n")
    elif isinstance(model_config, ConfigRecurrentLinearModel):
        item1 = "{0}: {1:}".format("MODEL_TYPE", config.MODEL_TYPE)
        item2 = "{0}: {1:}".format("NEURONS_PER_REPRESENTATION_LAYER", model_config.NEURONS_PER_REPRESENTATION_LAYER)
        print("{0:55} {1:55}".format(item1, item2, end="\n"))
        item1 = "{0}: {1:}".format("HIDDEN_SIZE", model_config.HIDDEN_SIZE)
        item2 = "{0}: {1:}".format("NUM_LAYERS", model_config.NUM_LAYERS)
        item3 = "{0}: {1:}".format("NEURONS_PER_FULLY_CONNECTED_LAYER", model_config.NEURONS_PER_FULLY_CONNECTED_LAYER)
        print("{0:55} {1:55} {2:55}".format(item1, item2, item3, end="\n"))
    elif isinstance(model_config, ConfigRecurrentConvolutionalModel):
        item1 = "{0}: {1:}".format("MODEL_TYPE", config.MODEL_TYPE)
        item2 = "{0}: {1:}".format("OUT_CHANNELS_PER_LAYER", model_config.OUT_CHANNELS_PER_LAYER)
        item3 = "{0}: {1:}".format("KERNEL_SIZE_PER_LAYER", model_config.KERNEL_SIZE_PER_LAYER)
        print("{0:55} {1:55} {2:55}".format(item1, item2, item3, end="\n"))
        item1 = "{0}: {1:}".format("STRIDE_PER_LAYER", model_config.STRIDE_PER_LAYER)
        item2 = "{0}: {1:}".format("PADDING", model_config.PADDING)
        item3 = "{0}: {1:}".format("NEURONS_PER_REPRESENTATION_LAYER", model_config.NEURONS_PER_REPRESENTATION_LAYER)
        print("{0:55} {1:55} {2:55}".format(item1, item2, item3, end="\n"))
        item1 = "{0}: {1:}".format("HIDDEN_SIZE", model_config.HIDDEN_SIZE)
        item2 = "{0}: {1:}".format("NUM_LAYERS", model_config.NUM_LAYERS)
        item3 = "{0}: {1:}".format("NEURONS_PER_FULLY_CONNECTED_LAYER", model_config.NEURONS_PER_FULLY_CONNECTED_LAYER)
        print("{0:55} {1:55} {2:55}".format(item1, item2, item3, end="\n"))
    else:
        raise ValueError()

    print("LAYER_ACTIVATION_TYPE: {0}".format(config.LAYER_ACTIVATION_TYPE))
    print("VALUE_NETWORK_LAYER_ACTIVATION_TYPE: {0}".format(config.VALUE_NETWORK_LAYER_ACTIVATION_TYPE))
    print("LOSS_FUNCTION_TYPE: {0}".format(config.LOSS_FUNCTION_TYPE))
    print("PLAY_MODEL_FILE_NAME: {0}".format(config.PLAY_MODEL_FILE_NAME))


def print_env_info(observation_space, action_space, config):
    # item1 = "{0}: {1:,}".format("EPISODE_REWARD_AVG_SOLVED", config.EPISODE_REWARD_AVG_SOLVED)
    # item2 = "{0}: {1:,}".format("EPISODE_REWARD_STD_SOLVED", config.EPISODE_REWARD_STD_SOLVED)
    # print("{0:55} {1:55}".format(item1, item2), end="\n")
    env_name_str = "ENV_NAME: {0}".format(config.ENV_NAME)
    print(env_name_str)

    observation_space_str = "OBSERVATION_SPACE: {0}, SHAPE: {1}".format(
        type(observation_space), observation_space.shape
    )
    print(observation_space_str)

    action_space_str = "ACTION_SPACE: {0}, SHAPE: {1}".format(
        type(action_space), action_space.shape
    )

    if isinstance(action_space, Discrete):
        action_space_str += ", N: {0}".format(action_space.n)
    elif isinstance(action_space, Box):
        action_bound_low, action_bound_high, action_scale, action_bias = get_continuous_action_info(
            action_space
        )
        action_space_str += ", LOW_BOUND: {0}, HIGH_BOUND: {1}, SCALE: {2}, BIAS: {3}".format(
            action_bound_low[0], action_bound_high[0], action_scale, action_bias
        )
    else:
        raise ValueError()
    print(action_space_str)


def console_log(
        total_episodes_v, last_mean_episode_reward_v,
        n_rollout_transitions_v, transition_rolling_rate_v, train_steps_v, train_step_rate_v,
        agent, config
):
    console_log = "[Tot. Episodes: {0:5,}] " \
                  "Mean Episode Reward: {1:6.1f}, Rolling Outs: {2:7,} ({3:7.3f}/sec.), " \
                  "Training Steps: {4:4,} ({5:.3f}/sec.), " \
        .format(
            total_episodes_v,
            last_mean_episode_reward_v,
            n_rollout_transitions_v,
            transition_rolling_rate_v,
            train_steps_v,
            train_step_rate_v
        )

    if config.AGENT_TYPE in [AgentType.DQN, AgentType.DUELING_DQN, AgentType.DOUBLE_DQN, AgentType.DOUBLE_DUELING_DQN]:
        console_log += "Q_net_loss: {0:>7.3f}, Epsilon: {1:>4.2f}".format(
            agent.last_q_net_loss.value, agent.epsilon.value
        )
    elif config.AGENT_TYPE == AgentType.REINFORCE:
        console_log += "log_policy_objective: {0:7.3f}".format(
            agent.last_log_policy_objective.value
        )
    elif config.AGENT_TYPE in (AgentType.A2C, AgentType.A3C):
        console_log += "critic_loss: {0:7.3f}, log_actor_obj.: {1:7.3f}, entropy: {2:5.3f}".format(
            agent.last_critic_loss.value, agent.last_actor_objective.value, agent.last_entropy.value
        )
    elif config.AGENT_TYPE in (AgentType.PPO, AgentType.PPO_TRAJECTORY):
        console_log += "critic_loss: {0:7.3f}, actor_obj.: {1:7.3f}, ratio: {2:5.3f}, entropy: {3:5.3f}".format(
            agent.last_critic_loss.value, agent.last_actor_objective.value, agent.last_ratio.value, agent.last_entropy.value
        )
    elif config.AGENT_TYPE == AgentType.SAC:
        console_log += "critic_loss: {0:7.3f}, actor_obj.: {1:7.3f}, alpha: {2:5.3f}, entropy: {3:5.3f}".format(
            agent.last_critic_loss.value, agent.last_actor_objective.value, agent.alpha.value, agent.last_entropy.value
        )
    elif config.AGENT_TYPE in (AgentType.DDPG, AgentType.TD3):
        console_log += "critic_loss: {0:7.3f}, actor_objective: {1:7.3f}".format(
            agent.last_critic_loss.value, agent.last_actor_objective.value
        )
    elif config.AGENT_TYPE == AgentType.MUZERO:
        console_log += "temperature: {0:7.3f}, value_loss: {1:7.3f}, policy_loss: {2:7.3f}, " \
                       "reward_loss: {3:7.3f}, total_loss: {4:7.3f}".format(
            agent.temperature.value, agent.value_loss.value, agent.policy_loss.value, agent.reward_loss.value,
            agent.loss.value
        )
    else:
        pass

    print(console_log)


def console_log_comparison(
        total_time_step, total_episodes_per_agent,
        last_mean_episode_reward_per_agent, n_rollout_transitions_per_agent, training_steps_per_agent,
        agents, config_c
):
    for agent_idx, agent in enumerate(agents):
        agent_prefix = "[Agent: {0}]".format(agent_idx)
        console_log = agent_prefix + "[Tot. Episodes: {0:5,}, Tot. Time Steps {1:7,}] " \
                      "Mean Episode Reward: {2:6.1f}, Rolling Outs: {3:7,}, " \
                      "Training Steps: {4:4,}, " \
            .format(
                total_episodes_per_agent[agent_idx],
                total_time_step,
                last_mean_episode_reward_per_agent[agent_idx],
                n_rollout_transitions_per_agent[agent_idx],
                training_steps_per_agent[agent_idx]
            )

        if config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE in [
            AgentType.DQN, AgentType.DOUBLE_DQN, AgentType.DUELING_DQN, AgentType.DOUBLE_DUELING_DQN
        ]:
            console_log += "Q_net_loss: {0:>6.3f}, Epsilon: {1:>4.2f}, ".format(
                agent.last_q_net_loss.value, agent.epsilon.value
            )
        elif config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.REINFORCE:
            console_log += "log_policy_objective: {0:6.3f}, ".format(
                agent.last_log_policy_objective.value
            )
        elif config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE in (AgentType.A2C, AgentType.A3C):
            console_log += "critic_loss: {0:6.3f}, log_actor_obj.: {1:5.3f}, ".format(
                agent.last_critic_loss.value, agent.last_actor_objective.value
            )
        elif config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE in (AgentType.PPO, AgentType.PPO_TRAJECTORY):
            console_log += "critic_loss: {0:7.3f}, actor_obj.: {1:7.3f}, ratio: {2:5.3f}, entropy: {3:5.3f}".format(
                agent.last_critic_loss.value, agent.last_actor_objective.value, agent.last_ratio.value, agent.last_entropy.value
            )
        elif config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.SAC:
            console_log += "critic_loss: {0:7.3f}, actor_obj.: {1:7.3f}, alpha: {2:5.3f}, entropy: {3:5.3f}".format(
                agent.last_critic_loss.value, agent.last_actor_objective.value, agent.alpha.value, agent.last_entropy.value
            )
        elif config_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE in (AgentType.DDPG, AgentType.TD3):
            console_log += "critic_loss: {0:7.3f}, actor_objective: {1:7.3f}, ".format(
                agent.last_critic_loss.value, agent.last_actor_objective.value
            )
        else:
            pass

        print(console_log)


def get_wandb_obj(config, agent=None, comparison=False):
    if comparison:
        project = "{0}_{1}_{2}".format(config.__class__.__name__, "Comparison", SYSTEM_USER_NAME)
    else:
        project = "{0}_{1}_{2}".format(config.ENV_NAME, config.AGENT_TYPE.name, SYSTEM_USER_NAME)

    wandb_obj = wandb.init(
        entity=config.WANDB_ENTITY,
        project=project,
        config={
            key: getattr(config, key) for key in dir(config) if not key.startswith("__")
        }
    )

    now = datetime.datetime.now()
    local_now = now.astimezone()
    wandb.run.name = local_now.strftime('%Y-%m-%d_%H:%M:%S')
    wandb.run.save()
    if agent:
        wandb.watch(agent.model, log="all")

    return wandb_obj


def wandb_log(learner, wandb_obj, config):
    log_dict = {
        "[TEST] Episode Reward": learner.test_episode_reward_avg.value,
        "[TEST] Std. of Episode Reward": learner.test_episode_reward_std.value,
        "Mean Episode Reward": learner.last_mean_episode_reward.value,
        "Episode": learner.total_episodes.value,
        "Buffer Size": len(learner.agent.buffer),
        "Training Steps": learner.training_step.value,
        "Total Time Steps": learner.total_time_step.value,
        "Transition Rolling Rate": learner.transition_rolling_rate.value,
        "Train Step Rate": learner.train_step_rate.value
    }

    if config.AGENT_TYPE in [AgentType.DQN, AgentType.DUELING_DQN, AgentType.DOUBLE_DQN, AgentType.DOUBLE_DUELING_DQN]:
        log_dict["QNet Loss"] = learner.agent.last_q_net_loss.value
        log_dict["Epsilon"] = learner.agent.epsilon.value
    elif config.AGENT_TYPE == AgentType.REINFORCE:
        log_dict["Log Policy Objective"] = learner.agent.last_log_policy_objective.value
    elif config.AGENT_TYPE in (AgentType.A2C, AgentType.A3C):
        log_dict["Critic Loss"] = learner.agent.last_critic_loss.value
        log_dict["Log Actor Objective"] = learner.agent.last_actor_objective.value
        log_dict["Entropy"] = learner.agent.last_entropy.value
    elif config.AGENT_TYPE in (AgentType.DDPG, AgentType.TD3):
        log_dict["Critic Loss"] = learner.agent.last_critic_loss.value
        log_dict["Actor Objective"] = learner.agent.last_actor_objective.value
    elif config.AGENT_TYPE in (AgentType.PPO, AgentType.PPO_TRAJECTORY):
        log_dict["Critic Loss"] = learner.agent.last_critic_loss.value
        log_dict["Actor Objective"] = learner.agent.last_actor_objective.value
        log_dict["Ratio"] = learner.agent.last_ratio.value
        log_dict["Entropy"] = learner.agent.last_entropy.value
    elif config.AGENT_TYPE == AgentType.SAC:
        log_dict["Critic Loss"] = learner.agent.last_critic_loss.value
        log_dict["Last Actor Objective"] = learner.agent.last_actor_objective.value
        log_dict["Alpha"] = learner.agent.alpha.value
        log_dict["Entropy"] = learner.agent.last_entropy.value
    elif config.AGENT_TYPE == AgentType.MUZERO:
        log_dict["Temperature"] = learner.agent.temperature.value
        log_dict["Value Loss"] = learner.agent.value_loss.value
        log_dict["Policy Loss"] = learner.agent.policy_loss.value
        log_dict["Reward Loss"] = learner.agent.reward_loss.value
        log_dict["Total Loss"] = learner.agent.loss.value

    else:
        pass

    if config.AGENT_TYPE in ActorCriticAgentTypes:
        log_dict["actor_grad_max"] = learner.agent.last_actor_model_grad_max.value
        log_dict["actor_grad_l2"] = learner.agent.last_actor_model_grad_l2.value
        log_dict["critic_grad_max"] = learner.agent.last_critic_model_grad_max.value
        log_dict["critic_grad_l2"] = learner.agent.last_critic_model_grad_l2.value
    elif config.AGENT_TYPE == AgentType.REINFORCE:
        log_dict["policy_grad_max"] = learner.agent.last_actor_model_grad_max.value
        log_dict["policy_grad_l2"] = learner.agent.last_actor_model_grad_l2.value
    else:
        log_dict["grad_max"] = learner.agent.last_model_grad_max.value
        log_dict["grad_l2"] = learner.agent.last_model_grad_l2.value

    wandb_obj.log(log_dict)


plotly_layout = go.Layout(
    plot_bgcolor="#FFF",  # Sets background color to white
    hovermode="x",
    hoverdistance=100, # Distance to show hover label of data point
    spikedistance=1000, # Distance to show spike
    xaxis=dict(
        title="Training Steps",
        linecolor="#BCCCDC",  # Sets color of X-axis line
        showgrid=True,
        showspikes=True, # Show spike line for X-axis
        # Format spike
        spikethickness=2,
        spikedash="dot",
        spikecolor="#999999",
        spikemode="across",
    ),
    yaxis=dict(
        # title="revenue",
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=True,
    ),
    legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=1),
    margin=dict(l=0.1, r=0.1, b=0.1, t=0.1),
    font=dict(size=10, color="Black")
)


# line_color_lst = ['green', 'red', 'blue', 'indigo', 'magenta']


def wandb_log_comparison(
        run, training_steps_per_agent, agents, agent_labels, n_episodes_for_mean_calculation, comparison_stat, wandb_obj
):
    training_steps_str = str([training_step for training_step in training_steps_per_agent])

    plotly_layout.yaxis.title = "[TEST] Episode Reward"
    plotly_layout.xaxis.title = "Training Steps ({0}, runs={1})".format(training_steps_str, run + 1)
    data = []
    for agent_idx, _ in enumerate(agents):
        data.append(
            go.Scatter(
                name=agent_labels[agent_idx],
                x=comparison_stat.test_training_steps_lst,
                y=comparison_stat.MEAN_test_episode_reward_avg_per_agent[agent_idx, :],
                showlegend=True
            )
        )
    test_episode_reward_avg = go.Figure(data=data, layout=plotly_layout)

    ###############################################################################
    plotly_layout.yaxis.title = "[TEST] Std. of Episode Reward"
    plotly_layout.xaxis.title = "Training Steps ({0}, runs={1})".format(training_steps_str, run + 1)
    data = []
    for agent_idx, _ in enumerate(agents):
        data.append(
            go.Scatter(
                name=agent_labels[agent_idx],
                x=comparison_stat.test_training_steps_lst,
                y=comparison_stat.MEAN_test_episode_reward_std_per_agent[agent_idx, :],
                showlegend=True
            )
        )
    test_episode_reward_std = go.Figure(data=data, layout=plotly_layout)

    ###############################################################################
    plotly_layout.yaxis.title = "[TRAIN] Mean Episode Reward"
    plotly_layout.xaxis.title = "Training Steps ({0}, runs={1}, over {2} Episodes)".format(
        training_steps_str, run + 1, n_episodes_for_mean_calculation
    )
    data = []
    for agent_idx, _ in enumerate(agents):
        data.append(
            go.Scatter(
                name=agent_labels[agent_idx],
                x=comparison_stat.test_training_steps_lst,
                y=comparison_stat.MEAN_mean_episode_reward_per_agent[agent_idx, :],
                showlegend=True
            )
        )
    train_last_mean_episode_reward = go.Figure(data=data, layout=plotly_layout)

    log_dict = {
        "episode_reward_avg": test_episode_reward_avg,
        "episode_reward_std": test_episode_reward_std,
        "train_last_mean_episode_reward": train_last_mean_episode_reward
    }

    wandb_obj.log(log_dict)


def get_train_env(config, no_graphics=True):
    def make_gym_env(env_name):
        def _make():
            if isinstance(config, ConfigUnityGymEnv):
                from sys import platform
                if platform == "linux" or platform == "linux2":
                    # linux
                    platform_dir = "linux"
                elif platform == "darwin":
                    # OS X
                    platform_dir = "mac"
                elif platform == "win32":
                    # Windows...
                    platform_dir = "windows"
                else:
                    raise ValueError()
                from gym_unity.envs import UnityToGymWrapper
                from mlagents_envs.environment import UnityEnvironment
                from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
                channel = EngineConfigurationChannel()
                u_env = UnityEnvironment(
                    file_name=os.path.join(config.UNITY_ENV_DIR, config.ENV_NAME, platform_dir,
                                           config.ENV_NAME),
                    worker_id=0, no_graphics=no_graphics, side_channels=[channel]
                )
                channel.set_configuration_parameters(time_scale=config.time_scale, width=config.width, height=config.height)
                env = UnityToGymWrapper(u_env)
                if config.ENV_NAME in ["UnityDrone"]:
                    from b_environments.wrappers.unity.unity_wrappers import GrayScaleObservation, ResizeObservation, TransformReward
                    env = gym.wrappers.FrameStack(ResizeObservation(GrayScaleObservation(env), shape=64), num_stack=4)
                    env = TransformReward(env)
                return env
            env = gym.make(env_name)
            if env_name in ["PongNoFrameskip-v4"]:
                env = gym.wrappers.AtariPreprocessing(
                    env, grayscale_obs=True, scale_obs=True
                )
                env = gym.wrappers.FrameStack(env, num_stack=4, lz4_compress=True)
            if env_name in ["FrozenLake-v1"]:
                env = MakeBoxFrozenLake()
            return env

        return _make

    train_env = AsyncVectorEnv(
        env_fns=[
            make_gym_env(config.ENV_NAME) for _ in range(config.N_VECTORIZED_ENVS)
        ]
    )

    return train_env


def get_single_env(config, no_graphics=True):
    if isinstance(config, ConfigUnityGymEnv):
        from sys import platform
        if platform == "linux" or platform == "linux2":
            # linux
            platform_dir = "linux"
        elif platform == "darwin":
            # OS X
            platform_dir = "mac"
        elif platform == "win32":
            # Windows...
            platform_dir = "windows"
        else:
            raise ValueError()

        from gym_unity.envs import UnityToGymWrapper
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        channel = EngineConfigurationChannel()
        u_env = UnityEnvironment(
            file_name=os.path.join(config.UNITY_ENV_DIR, config.ENV_NAME, platform_dir, config.ENV_NAME),
            worker_id=1, no_graphics=no_graphics, side_channels=[channel]
        )
        channel.set_configuration_parameters(time_scale=config.time_scale, width=config.width, height=config.height)
        single_env = UnityToGymWrapper(u_env)
        if config.ENV_NAME in ["UnityDrone"]:
            from b_environments.wrappers.unity.unity_wrappers import GrayScaleObservation, ResizeObservation, TransformReward
            single_env = gym.wrappers.FrameStack(ResizeObservation(GrayScaleObservation(single_env), shape=64), num_stack=4)
            single_env = TransformReward(single_env)
    else:
        if config.ENV_NAME in ["FrozenLake-v1"]:
            single_env = MakeBoxFrozenLake()
        else:
            single_env = gym.make(config.ENV_NAME)

            if config.ENV_NAME in ["PongNoFrameskip-v4"]:
                single_env = gym.wrappers.AtariPreprocessing(single_env, grayscale_obs=True, scale_obs=True)
                single_env = gym.wrappers.FrameStack(single_env, num_stack=4, lz4_compress=True)

    return single_env


# Box
# Dict
# Discrete
# MultiBinary
# MultiDiscrete
def get_env_info(config):
    single_env = get_single_env(config)

    observation_space = single_env.observation_space
    action_space = single_env.action_space

    single_env.close()

    return observation_space, action_space


def get_action_shape(action_space):
    n_discrete_actions = None
    action_shape = None

    if isinstance(action_space, Discrete):
        n_discrete_actions = (action_space.n,)
    elif isinstance(action_space, Box):
        action_shape = action_space.shape
    else:
        raise ValueError()

    return n_discrete_actions, action_shape


def get_continuous_action_info(action_space):
    action_bound_low = np.expand_dims(action_space.low, axis=0)
    action_bound_high = np.expand_dims(action_space.high, axis=0)

    assert np.equal(action_bound_high, -1.0 * action_bound_low).all()

    action_scale = (action_space.high - action_space.low) / 2.
    action_bias = (action_space.high + action_space.low) / 2.

    return action_bound_low, action_bound_high, action_scale, action_bias


def get_scaled_action():
    pass


class EpsilonTracker:
    def __init__(self, epsilon_init, epsilon_final, epsilon_final_training_step):
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_final_training_step = epsilon_final_training_step

    def epsilon(self, training_step):
        epsilon = max(
            self.epsilon_init - training_step / self.epsilon_final_training_step,
            self.epsilon_final
        )
        return epsilon


class MeanBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val):
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self):
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)
