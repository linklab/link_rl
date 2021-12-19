import time
import datetime

import gym
import torch
import os
import torch.multiprocessing as mp
import wandb
from gym.spaces import Discrete, Box
from gym.vector import AsyncVectorEnv
import plotly.graph_objects as go

from a_configuration.config.config import SYSTEM_USER_NAME
from g_utils.types import AgentType

if torch.cuda.is_available():
    import nvidia_smi
    nvidia_smi.nvmlInit()

    import pynvml
    pynvml.nvmlInit()
else:
    nvidia_smi = None
    pynvml = None


def model_save(model, env_name, agent_type_name, test_episode_reward_avg, test_episode_reward_std, parameter):
    env_model_home = os.path.join(parameter.MODEL_SAVE_DIR, env_name)
    if not os.path.exists(env_model_home):
        os.mkdir(env_model_home)

    agent_model_home = os.path.join(parameter.MODEL_SAVE_DIR, env_name, agent_type_name)
    if not os.path.exists(agent_model_home):
        os.mkdir(agent_model_home)

    now = datetime.datetime.now()
    local_now = now.astimezone()
    file_name = "{0:4.1f}_{1:3.1f}_{2}_{3}_{4}.pth".format(
        test_episode_reward_avg, test_episode_reward_std, local_now.year, local_now.month, local_now.day
    )

    torch.save(model.state_dict(), os.path.join(agent_model_home, file_name))


def model_load(model, env_name, agent_type_name, file_name, parameter):
    agent_model_home = os.path.join(parameter.MODEL_SAVE_DIR, env_name, agent_type_name)
    model_params = torch.load(os.path.join(agent_model_home, file_name))
    model.load_state_dict(model_params)


def print_device_related_info(device, parameter):
    n_cpu_cores = mp.cpu_count()
    print("{0:55} {1:55}".format(
        "DEVICE: {0}".format(device),
        "CPU CORES: {0}".format(n_cpu_cores),
    ), end="\n")
    print("{0:55} {1:55} {2:55}".format(
        "N_ACTORS: {0}".format(parameter.N_ACTORS),
        "ENVS PER ACTOR: {0}".format(parameter.N_VECTORIZED_ENVS),
        "TOTAL NUMBERS OF ENVS: {0}".format(
            parameter.N_ACTORS * parameter.N_VECTORIZED_ENVS
        )
    ))


def print_basic_info(observation_space=None, action_space=None, device=None, parameter=None):
    print('\n' + '#' * 72 + " Base Parameters " + '#' * 73)

    if device:
        print_device_related_info(device, parameter)
        print('-' * 75 + " Parameters " + '-' * 75)

    items = []

    for param in dir(parameter):
        if not param.startswith("__"):
            if param in [
                "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TRAINING_STEPS",
                "EPISODE_REWARD_AVG_SOLVED", "MAX_TRAINING_STEPS",
                "MIN_BUFFER_SIZE_FOR_TRAIN", "N_EPISODES_FOR_MEAN_CALCULATION",
                "TEST_INTERVAL_TRAINING_STEPS"
            ]:
                item = "{0}: {1:,}".format(param, getattr(parameter, param))
            else:
                item = "{0}: {1:}".format(param, getattr(parameter, param))
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

    if observation_space and action_space:
        if observation_space and action_space:
            print('-' * 76 + " SPACE " + '-' * 76)
        print_space(observation_space, action_space)

    print('#' * 162)
    print()


def print_comparison_basic_info(observation_space, action_space, device, parameter_c):
    print('\n' + '#' * 72 + " Base Parameters " + '#' * 73)

    if device:
        print_device_related_info(device, parameter_c)
        print('-' * 71 + " Common Parameters " + '-' * 71)

    items = []

    for param in dir(parameter_c):
        if param == "AGENT_LABELS":
            item1 = "{0}: {1:}".format("N_AGENTS", len(getattr(parameter_c, param)))
            item2 = "{0}: {1:}".format(param, getattr(parameter_c, param))
            print("{0:55} {1:55}".format(item1, item2))
            continue

        if param == "COMPARISON_RESULTS_SAVE_DIR":
            item = "{0}: {1:}".format(param, getattr(parameter_c, param))
            print("{0:55}".format(item))
            continue

        if param == "AGENT_PARAMETERS":
            continue

        if not param.startswith("__"):
            if param in [
                "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TRAINING_STEPS",
                "EPISODE_REWARD_AVG_SOLVED", "MAX_TRAINING_STEPS",
                "MIN_BUFFER_SIZE_FOR_TRAIN", "N_EPISODES_FOR_MEAN_CALCULATION",
                "TEST_INTERVAL_TRAINING_STEPS"
            ]:
                item = "{0}: {1:,}".format(param, getattr(parameter_c, param))
            else:
                item = "{0}: {1:}".format(param, getattr(parameter_c, param))
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

    for agent_idx, agent_parameter in enumerate(parameter_c.AGENT_PARAMETERS):
        print('-' * 76 + " Agent {0} ".format(agent_idx) + '-' * 76)
        for param in dir(agent_parameter):
            if not param.startswith("__"):
                if param in [
                    "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TRAINING_STEPS",
                    "EPISODE_REWARD_AVG_SOLVED", "MAX_TRAINING_STEPS",
                    "MIN_BUFFER_SIZE_FOR_TRAIN", "N_EPISODES_FOR_MEAN_CALCULATION",
                    "TEST_INTERVAL_TRAINING_STEPS"
                ]:
                    item = "{0}: {1:,}".format(param, getattr(agent_parameter, param))
                else:
                    item = "{0}: {1:}".format(param, getattr(agent_parameter, param))
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

    if observation_space and action_space:
        if observation_space and action_space:
            print('-' * 76 + " SPACE " + '-' * 76)
        print_space(observation_space, action_space)

    print('#' * 162)
    print()


def print_space(observation_space, action_space):
    observation_space_str = "OBSERVATION_SPACE: {0}, SHAPE: {1}".format(
        type(observation_space), observation_space.shape
    )
    print(observation_space_str)

    action_space_str = "ACTION_SPACE: {0}, SHAPE: {1}".format(
        type(action_space), action_space.shape
    )
    if isinstance(action_space, Discrete):
        action_space_str += ", N: {0}".format(action_space.n)
    print(action_space_str)


def console_log(
        total_train_start_time, total_episodes_v, total_time_steps_v,
        last_mean_episode_reward_v, n_rollout_transitions_v, train_steps_v,
        agent, parameter
):
    total_training_time = time.time() - total_train_start_time

    console_log = "[Total Episodes: {0:6,}, Total Time Steps {1:7,}] " \
                  "Mean Episode Reward: {2:5.1f}, Rolling Transitions: {3:7,} ({4:7.3f}/sec.), " \
                  "Training Steps: {5:5,} ({6:.3f}/sec.), " \
        .format(
            total_episodes_v,
            total_time_steps_v,
            last_mean_episode_reward_v,
            n_rollout_transitions_v,
            n_rollout_transitions_v / total_training_time,
            train_steps_v,
            train_steps_v / total_training_time
        )

    if parameter.AGENT_TYPE == AgentType.Dqn:
        console_log += "Q_net_loss: {0:>5.3f}, Epsilon: {1:>4.2f}, ".format(
            agent.last_q_net_loss.value, agent.epsilon.value
        )
    elif parameter.AGENT_TYPE == AgentType.Reinforce:
        console_log += "log_policy_objective: {0:5.1f}, ".format(
            agent.last_log_policy_objective.value
        )
    elif parameter.AGENT_TYPE == AgentType.A2c:
        console_log += "critic_loss: {0:5.1f}, log_actor_objective: {1:5.1f}, ".format(
            agent.last_critic_loss.value, agent.last_log_actor_objective.value
        )
    else:
        pass

    # if torch.cuda.is_available():
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # gpu_resource = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # console_log += f'gpu: {0}%, gpu-mem: {1}%, '.format(gpu_resource.gpu, gpu_resource.memory)
        # import GPUtil
        # gpu = GPUtil.getGPUs()[0]
        # console_log += f'gpu: {0}%, gpu-mem: {1}%, '.format(gpu.load * 100, gpu.memoryUtil * 100)

    print(console_log)


def console_log_comparison(
        total_time_steps, total_episodes_per_agent,
        last_mean_episode_reward_per_agent, n_rollout_transitions_per_agent, training_steps_per_agent,
        agents, parameter_c
):
    for agent_idx, agent in enumerate(agents):
        agent_prefix = "[Agent: {0}]".format(agent_idx)
        console_log = agent_prefix + "[Total Episodes: {0:6,}, Total Time Steps {1:7,}] " \
                      "Mean Episode Reward: {2:5.1f}, Rolling Transitions: {3:7,}, " \
                      "Training Steps: {4:5,}, " \
            .format(
                total_episodes_per_agent[agent_idx],
                total_time_steps,
                last_mean_episode_reward_per_agent[agent_idx],
                n_rollout_transitions_per_agent[agent_idx],
                training_steps_per_agent[agent_idx]
            )

        if parameter_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.Dqn:
            console_log += "Q_net_loss: {0:>5.3f}, Epsilon: {1:>4.2f}, ".format(
                agent.last_q_net_loss.value, agent.epsilon.value
            )
        elif parameter_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.Reinforce:
            console_log += "log_policy_objective: {0:5.3f}, ".format(
                agent.last_log_policy_objective.value
            )
        elif parameter_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.A2c:
            console_log += "critic_loss: {0:5.3f}, log_actor_objective: {1:5.3f}, ".format(
                agent.last_critic_loss.value, agent.last_log_actor_objective.value
            )
        else:
            pass

        print(console_log)


def get_wandb_obj(parameter, agent=None, comparison=False):
    if comparison:
        project = "{0}_{1}_{2}".format(parameter.ENV_NAME, "Comparison", SYSTEM_USER_NAME)
    else:
        project = "{0}_{1}_{2}".format(parameter.ENV_NAME, parameter.AGENT_TYPE.name, SYSTEM_USER_NAME)

    wandb_obj = wandb.init(
        entity=parameter.WANDB_ENTITY,
        project=project,
        config={
            key: getattr(parameter, key) for key in dir(parameter) if not key.startswith("__")
        }
    )

    now = datetime.datetime.now()
    local_now = now.astimezone()
    wandb.run.name = local_now.strftime('%Y-%m-%d_%H:%M:%S')
    wandb.run.save()
    wandb.watch(agent.model, log="all")

    return wandb_obj


def wandb_log(learner, wandb_obj, parameter):
    log_dict = {
        "[TEST] Episode Reward": learner.test_episode_reward_avg.value,
        "[TEST] Std. of Episode Reward": learner.test_episode_reward_std.value,
        "Mean Episode Reward": learner.last_mean_episode_reward.value,
        "Episode": learner.total_episodes.value,
        "Buffer Size": learner.agent.buffer.size(),
        "Training Steps": learner.training_steps.value,
        "Total Time Steps": learner.total_time_steps.value
    }

    if parameter.AGENT_TYPE == AgentType.Dqn:
        log_dict["QNet Loss"] = learner.agent.last_q_net_loss.value
        log_dict["Epsilon"] = learner.agent.epsilon.value
    elif parameter.AGENT_TYPE == AgentType.Reinforce:
        log_dict["Log Policy Objective"] = learner.agent.last_log_policy_objective.value
    elif parameter.AGENT_TYPE == AgentType.A2c:
        log_dict["Critic Loss"] = learner.agent.last_critic_loss.value
        log_dict["Log Actor Objective"] = learner.agent.last_log_actor_objective.value
    else:
        pass

    wandb_obj.log(log_dict)


# def wandb_log_comparison(learner, wandb_obj):
#     log_dict = {
#         "[TEST] Episode Reward": learner.test_episode_reward_avg.value,
#         "[TEST] Std. of Episode Reward": learner.test_episode_reward_std.value,
#         "Mean Episode Reward": learner.last_mean_episode_reward.value,
#         "Episode": learner.total_episodes.value,
#         "Buffer Size": learner.n_rollout_transitions.value,
#         "Training Steps": learner.training_steps.value,
#         "Total Time Steps": learner.total_time_steps.value
#     }
#     wandb_obj.log(log_dict)

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
        run, agents, agent_labels, n_episodes_for_mean_calculation, comparison_stat, wandb_obj
):
    plotly_layout.yaxis.title = "[TEST] Episode Reward"
    plotly_layout.xaxis.title = "Training Steps (runs={0})".format(run + 1)
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
    plotly_layout.xaxis.title = "Training Steps (runs={0})".format(run + 1)
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
    plotly_layout.xaxis.title = "Training Steps (Recent {0} Episodes, runs={1})".format(
        n_episodes_for_mean_calculation, run + 1
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

# def wandb_log_comparison(agents, agent_labels, comparison_stat, wandb_obj):
#     plotly_layout.yaxis.title = "[TEST] Episode Reward"
#
#     test_episode_reward_avg = go.Figure(layout=plotly_layout)
#
#     test_episode_reward_avg.add_traces(
#         go.Scatter(
#             x=comparison_stat.test_training_steps_lst,
#             y=comparison_stat.MAX_test_episode_reward_avg_per_agent[0, :],
#             fill=None,
#             line_color=line_color_lst[0],
#         )
#     )
#
#     test_episode_reward_avg.add_traces(
#         go.Scatter(
#             x=comparison_stat.test_training_steps_lst,
#             y=comparison_stat.MIN_test_episode_reward_avg_per_agent[0, :],
#             fill="tonexty",
#             fillcolor="rgba(250, 0, 0, 0.4)",
#             line_color=line_color_lst[0],
#         )
#     )

    # data = []
    # for agent_idx, _ in enumerate(agents):
    #     data.append(
    #         go.Scatter(
    #             name=agent_labels[agent_idx],
    #             x=comparison_stat.test_training_steps_lst,
    #             y=comparison_stat.MAX_test_episode_reward_avg_per_agent[agent_idx, :],
    #             fill=None,
    #             mode="lines",
    #             line_color=line_color_lst[agent_idx],
    #             showlegend=False
    #         )
    #     )
    #     data.append(
    #         go.Scatter(
    #             name=agent_labels[agent_idx],
    #             x=comparison_stat.test_training_steps_lst,
    #             y=comparison_stat.MIN_test_episode_reward_avg_per_agent[agent_idx, :],
    #             fill="tonexty",
    #             mode="lines",
    #             line_color=line_color_lst[agent_idx],
    #             showlegend=False
    #         )
    #     )
    #     # data.append(
    #     #     go.Scatter(
    #     #         name=agent_labels[agent_idx],
    #     #         x=comparison_stat.test_training_steps_lst,
    #     #         y=comparison_stat.MEAN_test_episode_reward_avg_per_agent[agent_idx, :],
    #     #         mode="lines",
    #     #         line_color=line_color_lst[agent_idx],
    #     #         showlegend=True
    #     #     )
    #     # )
    #     if agent_idx == 0:
    #         break
    #
    # test_episode_reward_avg = go.Figure(data=data, layout=plotly_layout)

    ###############################################################################
    # plotly_layout.yaxis.title = "[TEST] Std. of Episode Reward"
    # data = []
    # for agent_idx, _ in enumerate(agents):
    #     data.append(
    #         go.Scatter(
    #             name=agent_labels[agent_idx],
    #             x=comparison_stat.test_training_steps_lst,
    #             y=comparison_stat.MAX_test_episode_reward_std_per_agent[agent_idx, :],
    #             fill=None,
    #             mode="lines",
    #             line_color=line_color_lst[agent_idx],
    #             showlegend=False
    #         )
    #     )
    #     data.append(
    #         go.Scatter(
    #             name=agent_labels[agent_idx],
    #             x=comparison_stat.test_training_steps_lst,
    #             y=comparison_stat.MIN_test_episode_reward_std_per_agent[agent_idx, :],
    #             fill="tonexty",
    #             mode="lines",
    #             line_color=line_color_lst[agent_idx],
    #             showlegend=False
    #         )
    #     )
    #     data.append(
    #         go.Scatter(
    #             name=agent_labels[agent_idx],
    #             x=comparison_stat.test_training_steps_lst,
    #             y=comparison_stat.MEAN_test_episode_reward_std_per_agent[agent_idx, :],
    #             mode="lines",
    #             line_color=line_color_lst[agent_idx],
    #             showlegend=True
    #         )
    #     )
    # test_episode_reward_std = go.Figure(data=data, layout=plotly_layout)
    #
    # ###############################################################################
    # plotly_layout.yaxis.title = "Last Mean Episode Reward"
    # data = []
    # for agent_idx, _ in enumerate(agents):
    #     data.append(
    #         go.Scatter(
    #             name=agent_labels[agent_idx],
    #             x=comparison_stat.test_training_steps_lst,
    #             y=comparison_stat.MAX_mean_episode_reward_per_agent[agent_idx, :],
    #             fill=None,
    #             mode="lines",
    #             line_color=line_color_lst[agent_idx],
    #             showlegend=False
    #         )
    #     )
    #     data.append(
    #         go.Scatter(
    #             name=agent_labels[agent_idx],
    #             x=comparison_stat.test_training_steps_lst,
    #             y=comparison_stat.MIN_mean_episode_reward_per_agent[agent_idx, :],
    #             fill="tonexty",
    #             mode="lines",
    #             line_color=line_color_lst[agent_idx],
    #             showlegend=False
    #         )
    #     )
    #     data.append(
    #         go.Scatter(
    #             name=agent_labels[agent_idx],
    #             x=comparison_stat.test_training_steps_lst,
    #             y=comparison_stat.MEAN_mean_episode_reward_per_agent[agent_idx, :],
    #             mode="lines",
    #             line_color=line_color_lst[agent_idx],
    #             showlegend=True
    #         )
    #     )
    # train_last_mean_episode_reward = go.Figure(data=data, layout=plotly_layout)
    #
    # log_dict = {
    #     "episode_reward_avg": test_episode_reward_avg,
    #     "episode_reward_std": test_episode_reward_std,
    #     "train_last_mean_episode_reward": train_last_mean_episode_reward
    # }
    #
    # wandb_obj.log(log_dict)


def get_train_env(parameter):
    def make_gym_env(env_name):
        def _make():
            env = gym.make(env_name)
            if env_name in ["PongNoFrameskip-v4"]:
                env = gym.wrappers.AtariPreprocessing(
                    env, grayscale_obs=True, scale_obs=True
                )
                env = gym.wrappers.FrameStack(env, num_stack=4, lz4_compress=True)
            return env

        return _make

    train_env = AsyncVectorEnv(
        env_fns=[
            make_gym_env(parameter.ENV_NAME) for _ in range(parameter.N_VECTORIZED_ENVS)
        ]
    )

    return train_env


def get_single_env(parameter):
    single_env = gym.make(parameter.ENV_NAME)
    if parameter.ENV_NAME in ["PongNoFrameskip-v4"]:
        single_env = gym.wrappers.AtariPreprocessing(
            single_env, grayscale_obs=True, scale_obs=True
        )
        single_env = gym.wrappers.FrameStack(single_env, num_stack=4, lz4_compress=True)

    return single_env


def get_test_env(params):
    test_env = gym.make(params.ENV_NAME)
    if params.ENV_NAME in ["PongNoFrameskip-v4"]:
        test_env = gym.wrappers.AtariPreprocessing(
            test_env, grayscale_obs=True, scale_obs=True
        )
        test_env = gym.wrappers.FrameStack(test_env, num_stack=4, lz4_compress=True)

    observation_shape = test_env.observation_space.shape
    n_actions = test_env.action_space.n

    return test_env, observation_shape, n_actions


# Box
# Dict
# Discrete
# MultiBinary
# MultiDiscrete
def get_env_info(parameter):
    single_env = get_single_env(parameter)

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

