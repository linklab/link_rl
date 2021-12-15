import time
from datetime import date

import gym
import torch
import os
import torch.multiprocessing as mp
import wandb
from gym.vector import AsyncVectorEnv

from a_configuration.config.config import Config
from g_utils.types import AgentType

if torch.cuda.is_available():
    import nvidia_smi
    nvidia_smi.nvmlInit()

    import pynvml
    pynvml.nvmlInit()
else:
    nvidia_smi = None
    pynvml = None



def model_save(model, env_name, agent_type_name, test_episode_reward_avg, test_episode_reward_std):
    env_model_home = os.path.join(Config.MODEL_HOME, env_name)
    if not os.path.exists(env_model_home):
        os.mkdir(env_model_home)

    agent_model_home = os.path.join(Config.MODEL_HOME, env_name, agent_type_name)
    if not os.path.exists(agent_model_home):
        os.mkdir(agent_model_home)

    today_date = date.today()

    file_name = "{0:4.1f}_{1:3.1f}_{2}_{3}_{4}.pth".format(
        test_episode_reward_avg, test_episode_reward_std,
        today_date.year, today_date.month, today_date.day
    )

    torch.save(model.state_dict(), os.path.join(agent_model_home, file_name))


def model_load(model, env_name, agent_type_name, file_name):
    agent_model_home = os.path.join(Config.MODEL_HOME, env_name, agent_type_name)
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


def print_basic_info(device, parameter):
    print('\n' + '#' * 72 + " Base Parameters " + '#' * 73)

    if device:
        print_device_related_info(device, parameter)
        print('-' * 75 + " Parameters " + '-' * 75)

    items = []

    for param in dir(parameter):
        if not param.startswith("__"):
            if param in [
                "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS",
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

    print('#' * 162)
    print()


def print_comparison_basic_info(device, parameter):
    print('\n' + '#' * 72 + " Base Parameters " + '#' * 73)

    if device:
        print_device_related_info(device, parameter)
        print('-' * 75 + " Parameters " + '-' * 75)

    items = []

    for param in dir(parameter):
        if param == "AGENT_LABELS":
            item1 = "{0}: {1:}".format("N_AGENTS", len(getattr(parameter, param)))
            item2 = "{0}: {1:}".format(param, getattr(parameter, param))
            print("{0:55} {1:55}".format(item1, item2))
            continue

        if param == "AGENT_PARAMETERS":
            continue

        if not param.startswith("__"):
            if param in [
                "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS",
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

    for agent_idx, agent_parameter in enumerate(parameter.AGENT_PARAMETERS):
        print('-' * 75 + " Agent {0} ".format(agent_idx) + '-' * 75)
        for param in dir(agent_parameter):
            if not param.startswith("__"):
                if param in [
                    "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS",
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

    print('#' * 162)
    print()


def console_log(
        total_train_start_time, total_episodes_v, total_time_steps_v,
        last_mean_episode_reward_v, n_rollout_transitions_v, train_steps_v,
        agent, parameter
):
    total_training_time = time.time() - total_train_start_time
    formatted_total_training_time = time.strftime(
        '%H:%M:%S', time.gmtime(total_training_time)
    )

    console_log = "[Total Episodes: {0:5,}, Total Time Steps {1:7,}] " \
                  "Mean Episode Reward: {2:5.1f}, Rolling Transitions: {3:6,} ({4:7.3f}/sec.), " \
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
        console_log += "Q_net_loss: {0:5.1f}, Epsilon: {0:4.2f}, ".format(
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

    console_log += "Total Elapsed Time {}".format(formatted_total_training_time)

    print(console_log)


def console_log_comparison(
        total_time_steps, total_episodes_per_agent,
        last_mean_episode_reward_per_agent, n_rollout_transitions_per_agent, training_steps_per_agent,
        agents, parameter_c
):
    for agent_idx, agent in enumerate(agents):
        agent_prefix = "[Agent: {0}]".format(agent_idx)
        console_log = agent_prefix + "[Total Episodes: {0:5,}, Total Time Steps {1:7,}] " \
                      "Mean Episode Reward: {2:5.1f}, Rolling Transitions: {3:6,}, " \
                      "Training Steps: {4:5,}, " \
            .format(
                total_episodes_per_agent[agent_idx],
                total_time_steps,
                last_mean_episode_reward_per_agent[agent_idx],
                n_rollout_transitions_per_agent[agent_idx],
                training_steps_per_agent[agent_idx]
            )

        if parameter_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.Dqn:
            console_log += "Q_net_loss: {0:5.1f}, Epsilon: {0:4.2f}, ".format(
                agent.last_q_net_loss.value, agent.epsilon.value
            )
        elif parameter_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.Reinforce:
            console_log += "log_policy_objective: {0:5.1f}, ".format(
                agent.last_log_policy_objective.value
            )
        elif parameter_c.AGENT_PARAMETERS[agent_idx].AGENT_TYPE == AgentType.A2c:
            console_log += "critic_loss: {0:5.1f}, log_actor_objective: {1:5.1f}, ".format(
                agent.last_critic_loss.value, agent.last_log_actor_objective.value
            )
        else:
            pass

        print(console_log)


def get_wandb_obj(parameter, comparison=False):
    project = "{0}_{1}".format(parameter.ENV_NAME, "Comparison") \
        if comparison else "{0}_{1}".format(parameter.ENV_NAME, parameter.AGENT_TYPE.name)

    wandb_obj = wandb.init(
        entity=parameter.WANDB_ENTITY,
        project=project,
        config={
            key: getattr(parameter, key) for key in dir(parameter) if not key.startswith("__")
        }
    )
    return wandb_obj


def wandb_log(learner, wandb_obj, parameter):
    log_dict = {
        "[TEST] Average Episode Reward": learner.test_episode_reward_avg.value,
        "[TEST] Std. Episode Reward": learner.test_episode_reward_std.value,
        "Mean Episode Reward": learner.last_mean_episode_reward.value,
        "Episode": learner.total_episodes.value,
        "Buffer Size": learner.n_rollout_transitions.value,
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
#         "[TEST] Average Episode Reward": learner.test_episode_reward_avg.value,
#         "[TEST] Std. Episode Reward": learner.test_episode_reward_std.value,
#         "Mean Episode Reward": learner.last_mean_episode_reward.value,
#         "Episode": learner.total_episodes.value,
#         "Buffer Size": learner.n_rollout_transitions.value,
#         "Training Steps": learner.training_steps.value,
#         "Total Time Steps": learner.total_time_steps.value
#     }
#     wandb_obj.log(log_dict)

import plotly.graph_objects as go

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
    margin=dict(l=0.1, r=0.1, b=0.1, t=0.1)
)


def wandb_log_comparison(learner, wandb_obj, test_idx):
    print("$$$$$$$$")
    print(learner.MEAN_test_episode_reward_avg_per_agent[0, :test_idx + 1])
    print(learner.MEAN_test_episode_reward_avg_per_agent[:, :test_idx + 1])
    print("$$$$$$$$")
    ###############################################################################
    plotly_layout.yaxis.title = "[TEST] Average Episode Reward"
    test_episode_reward_avg = go.Figure(
        data=[
            go.Scatter(
                name=learner.parameter_c.AGENT_LABELS[agent_idx],
                x=learner.test_training_steps_lst,
                y=learner.MEAN_test_episode_reward_avg_per_agent[agent_idx, :test_idx + 1],
                mode="lines",
                showlegend=True
            ) for agent_idx, _ in enumerate(learner.agents)
        ],
        layout=plotly_layout
    )

    test_episode_reward_avg_2 = wandb.plot.line_series(
        xs=learner.test_training_steps_lst,
        ys=learner.MEAN_test_episode_reward_avg_per_agent[:, :test_idx + 1],
        keys=learner.parameter_c.AGENT_LABELS,
        title="[TEST] Average Episode Reward", xname="Training_Steps"
    )

    ###############################################################################
    plotly_layout.yaxis.title = "[TEST] Std. Episode Reward"
    test_episode_reward_std = go.Figure(
        data=[
            go.Scatter(
                name=learner.parameter_c.AGENT_LABELS[agent_idx],
                x=learner.test_training_steps_lst,
                y=learner.MEAN_test_episode_reward_std_per_agent[agent_idx, :test_idx + 1],
                mode="lines",
                showlegend=True
            ) for agent_idx, _ in enumerate(learner.agents)
        ],
        layout=plotly_layout
    )

    test_episode_reward_std_2 = wandb.plot.line_series(
        xs=learner.test_training_steps_lst,
        ys=learner.MEAN_test_episode_reward_std_per_agent[:, :test_idx + 1],
        keys=learner.parameter_c.AGENT_LABELS,
        title="[TEST] Std. Episode Reward", xname="Training_Steps"
    )

    ###############################################################################
    plotly_layout.yaxis.title = "Last Mean Episode Reward"
    train_last_mean_episode_reward = go.Figure(
        data=[
            go.Scatter(
                name=learner.parameter_c.AGENT_LABELS[agent_idx],
                x=learner.test_training_steps_lst,
                y=learner.MEAN_mean_episode_reward_per_agent[agent_idx, :test_idx + 1],
                mode="lines",
                showlegend=True
            ) for agent_idx, _ in enumerate(learner.agents)
        ],
        layout=plotly_layout
    )

    train_last_mean_episode_reward_2 = wandb.plot.line_series(
        xs=learner.test_training_steps_lst,
        ys=learner.MEAN_mean_episode_reward_per_agent[:, :test_idx + 1],
        keys=learner.parameter_c.AGENT_LABELS,
        title="Last Mean Episode Reward", xname="Training_Steps"
    )

    log_dict = {
        "episode_reward_avg": test_episode_reward_avg,
        "episode_reward_std": test_episode_reward_std,
        "train_last_mean_episode_reward": train_last_mean_episode_reward,
        "episode_reward_avg_2": test_episode_reward_avg_2,
        "episode_reward_std_2": test_episode_reward_std_2,
        "train_last_mean_episode_reward_2": train_last_mean_episode_reward_2
    }

    wandb_obj.log(log_dict)

    # log_dict = {
    #     "episode_reward_avg": wandb.plot.line_series(
    #         xs=learner.test_training_steps_lst,
    #         ys=learner.test_episode_reward_avg_per_agent, keys=learner.parameter_c.AGENT_LABELS,
    #         title="[TEST] Average Episode Reward", xname="Training Steps"
    #     ),
    #     "episode_reward_std": wandb.plot.line_series(
    #         xs=learner.test_training_steps_lst,
    #         ys=learner.test_episode_reward_std_per_agent, keys=learner.parameter_c.AGENT_LABELS,
    #         title="[TEST] Std. Episode Reward", xname="Training Steps"
    #     ),
    # }
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

    if parameter.ENV_NAME in ["CartPole-v1", "PongNoFrameskip-v4"]:
        train_env = AsyncVectorEnv(
            env_fns=[
                make_gym_env(parameter.ENV_NAME) for _ in range(parameter.N_VECTORIZED_ENVS)
            ]
        )
    else:
        raise ValueError()

    return train_env


def get_single_env(parameter):
    single_env = gym.make(parameter.ENV_NAME)
    if parameter.ENV_NAME in ["PongNoFrameskip-v4"]:
        single_env = gym.wrappers.AtariPreprocessing(
            single_env, grayscale_obs=True, scale_obs=True
        )
        single_env = gym.wrappers.FrameStack(single_env, num_stack=4, lz4_compress=True)

    return single_env


def get_env_info(parameter):
    single_env = get_single_env(parameter)

    obs_shape = single_env.observation_space.shape
    n_actions = single_env.action_space.n

    single_env.close()
    return obs_shape, n_actions


class EpsilonTracker:
    def __init__(self, epsilon_init, epsilon_final, epsilon_final_time_step):
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_final_time_step = epsilon_final_time_step

    def epsilon(self, training_step):
        epsilon = max(
            self.epsilon_init - training_step / self.epsilon_final_time_step,
            self.epsilon_final
        )
        return epsilon
